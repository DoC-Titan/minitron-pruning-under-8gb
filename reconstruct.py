"""Reconstruct a retained adapter+physical recipe without training or downloads."""
import argparse
from core import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise RuntimeError('Existing output protected')
    configure(a.device);job=next(j for j in jobs() if j['id']==a.job)
    model,_=load_model(a.device);recipe=json.loads(recipe_path(job).read_text());apply_recipe(model,recipe)
    state=checked_load(ROOT/'adapters'/f'{a.job}.pt')
    with torch.no_grad():
        for i,layer in enumerate(model.model.layers):
            for key in ['q_proj','v_proj']:
                prefix=f'model.layers.{i}.self_attn.{key}';w=getattr(layer.self_attn,key).weight
                delta=state[prefix+'.B']@state[prefix+'.A']*C['alpha']/C['rank']
                w.copy_((w.detach().half().cpu()+delta.half()).half().to(w.device,dtype=w.dtype))
    model.save_pretrained(a.output,safe_serialization=True,max_shard_size='2GB');tokenizer().save_pretrained(a.output)
    shutil.copy2(ROOT/'licenses/QWEN_RESEARCH_LICENSE.txt',a.output/'LICENSE.txt')
    (a.output/'NOTICE.txt').write_text((ROOT/'NOTICE.txt').read_text(encoding='utf-8').split('\nSupplied NVIDIA')[0].replace('licenses/QWEN_RESEARCH_LICENSE.txt','LICENSE.txt'),encoding='utf-8')
    atomic_json(a.output/'pruning_recipe.json',recipe)
    print('Reconstructed',a.job,'at',a.output)


if __name__=='__main__':main()
