"""Executed reduced CPU/CUDA pruning+KD study; fixed byte-level corpus, FP32."""
import argparse
import copy
from core import *
from tests import tiny_model


def blocks(which):
    x=torch.load(ROOT/'data/byte_corpus.pt',weights_only=True)[which]
    return x[:(len(x)//64)*64].reshape(-1,64)


def fit(model,device,seed,key,teacher=None,objective='ce',steps=128):
    path=ROOT/'state/tiny'/f'{key}_{device}_s{seed}.pt'
    state=checked_load(path,backup=True) if path.exists() else None
    model.requires_grad_(True);model.to(device);model.train();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.)
    train=blocks('train');gen=torch.Generator().manual_seed(seed+700)
    indices=torch.randint(0,len(train),(steps,4),generator=gen)
    start=0;elapsed=0.;losses=[]
    if state:
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);start=state['step'];elapsed=state['seconds'];losses=state['losses']
    initial=nll(model,blocks('validation')[:16],device) if start==0 else state['initial']
    for step in range(start,steps):
        guard(device);t=time.perf_counter();x=train[indices[step]].to(device);opt.zero_grad(set_to_none=True);model.train()
        z=model(input_ids=x,use_cache=False).logits[:,:-1]
        if objective=='kl':
            with torch.no_grad():target=teacher(input_ids=x,use_cache=False).logits[:,:-1]
            loss=full_kl(z,target)
        else:loss=F.cross_entropy(z.reshape(-1,256),x[:,1:].reshape(-1))
        assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        assert torch.isfinite(norm);opt.step();sync(device);elapsed+=time.perf_counter()-t;losses.append(float(loss))
        atomic_torch(path,{'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':opt.state_dict(),
                          'step':step+1,'seconds':elapsed,'losses':losses,'initial':initial},backup=True)
        if (ROOT/'STOP').exists():raise RuntimeError('USER PAUSE: tiny checkpoint saved')
    model.eval()
    return {'initial_validation':initial,'final_validation':nll(model,blocks('validation')[:32],device),
            'test':nll(model,blocks('test')[:32],device),'training_seconds':elapsed,'losses':losses,
            'input_tokens':steps*4*64,'target_tokens':steps*4*63,'updates':steps,'memory':memory(device)}


def prune_teacher(teacher,device,seed,criterion):
    model=copy.deepcopy(teacher);original=list(model.model.layers);cal=blocks('train')[-8:];ranks=[]
    for idx in [1,2]:
        model.model.layers=nn.ModuleList([l for i,l in enumerate(original) if i!=idx]);fix_layers(model)
        ranks.append({'layer':idx,'nll':nll(model,cal,device)['nll']})
    removed=random.Random(seed).choice([1,2]) if criterion=='random' else min(ranks,key=lambda r:r['nll'])['layer']
    model.model.layers=nn.ModuleList(original);fix_layers(model);kept=[i for i in range(4) if i!=removed];channels=[]
    for i in kept:
        mlp=model.model.layers[i].mlp
        score=(mlp.gate_proj.weight.detach().square().sum(1)+mlp.up_proj.weight.detach().square().sum(1)+mlp.down_proj.weight.detach().square().sum(0)).sqrt().cpu()
        ids=(torch.randperm(192,generator=torch.Generator().manual_seed(seed+i))[:128].sort().values if criterion=='random' else torch.argsort(score,descending=True,stable=True)[:128].sort().values)
        channels.append({'original_layer':i,'retained':ids.tolist()})
    recipe={'retained_layers':kept,'removed_layers':[removed],'ffn':128,'channels':channels,'calibration':ranks}
    apply_recipe(model,recipe);return model,recipe


@torch.inference_mode()
def benchmark_same_weights(device):
    # CPU-trained state used on BOTH devices: isolates device rather than training effects.
    saved=checked_load(ROOT/'state/tiny/informed_kl_cpu_s42.pt')
    model=tiny_model();apply_recipe(model,{'retained_layers':[0,1,3],'ffn':128,'channels':[{'retained':list(range(128))} for _ in range(3)]})
    # Shapes match; actual learned structural recipe is in the metric file.
    model.load_state_dict(saved['model']);model.to(device);model.eval();cases=[]
    for length in [1,8,32,128,512]:
        host=blocks('test').reshape(-1)[:length].unsqueeze(0)
        def request():
            t=time.perf_counter();x=host.to(device);sync(device);transfer=time.perf_counter()-t;t=time.perf_counter()
            y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),min_new_tokens=16,max_new_tokens=16,
                             do_sample=False,use_cache=True,pad_token_id=0,eos_token_id=None)
            sync(device);execution=time.perf_counter()-t
            return {'execution_seconds':execution,'transfer_seconds':transfer,'request_e2e_seconds':execution+transfer}
        for _ in range(2):request()
        runs=[request() for _ in range(5)]
        cases.append({'prompt_tokens':length,'new_tokens':16,'runs':runs,'median_seconds':statistics.median(r['execution_seconds'] for r in runs),
                      'median_e2e_seconds':statistics.median(r['request_e2e_seconds'] for r in runs)})
    return {'device':device,'dtype':'float32','matched_weights_sha256':sha(ROOT/'state/tiny/informed_kl_cpu_s42.pt'),
            'quality':nll(model,blocks('test')[:32],device),'cases':cases,'memory':memory(device),
            'scope':'same tiny CPU-trained weights, FP32 on both devices; excludes model loading; fixed byte inputs, no tokenization',
            'warmups':2,'timing_repeats':5}


def run(device):
    output=ROOT/'metrics'/f'tiny_{device}.json'
    if output.exists():return
    rows=[]
    for seed in C['seeds']:
        seed_all(seed);teacher=tiny_model(seed);teacher.config.use_cache=False
        info=fit(teacher,device,seed,'teacher',steps=128);teacher.requires_grad_(False);teacher.eval()
        rows.append({'seed':seed,'criterion':'teacher','objective':'ce','parameters':sum(p.numel() for p in teacher.parameters()),'result':info})
        for criterion in C['criteria']:
            initial,recipe=prune_teacher(teacher,device,seed,criterion)
            for objective in ['kl','ce']:
                model=copy.deepcopy(initial);seed_all(seed)
                result=fit(model,device,seed,criterion+'_'+objective,teacher=teacher,objective=objective,steps=64)
                rows.append({'seed':seed,'criterion':criterion,'objective':objective,'parameters':sum(p.numel() for p in model.parameters()),'recipe':recipe,'result':result})
                del model
        atomic_json(output.with_suffix('.partial.json'),{'rows':rows})
        print('TINY',device,seed,'complete',flush=True)
    atomic_json(output,{'device':device,'dtype':'float32','cpu_threads':torch.get_num_threads(),'rows':rows,
                       'benchmark':benchmark_same_weights(device),'notes':['Trained from scratch, byte vocabulary256; correctness/reference only.',
                       'Not a pretrained sub-billion replacement or a 3B robustness result.','CPU and GPU teacher/student training use the same seeds, architecture, batch indices and FP32.']})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],required=True);args=p.parse_args()
    configure(args.device);run(args.device)
