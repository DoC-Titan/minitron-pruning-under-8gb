"""Short local-only completion demo. No training, downloads or benchmark claims."""
import argparse
from core import *


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--prompt',default='Machine learning is a branch of artificial intelligence that')
    p.add_argument('--new-tokens',type=int,default=32);a=p.parse_args()
    if not 1<=a.new_tokens<=64:p.error('Keep demo output between1and64tokens')
    configure(a.device)
    assert (ROOT/'model/EXPORT_COMPLETE.json').exists(),'No completed local export. Read REPRODUCE.md.'
    tok=tokenizer();ids=tok(a.prompt,return_tensors='pt')
    if ids.input_ids.shape[1]>128:p.error('Demo prompt exceeds128tokens')
    model,_=load_model(a.device,ROOT/'model')
    with torch.inference_mode():out=model.generate(**ids.to(a.device),max_new_tokens=a.new_tokens,do_sample=False,pad_token_id=tok.eos_token_id)
    print(tok.decode(out[0],skip_special_tokens=True))
    print('Actual device:',next(model.parameters()).device,'dtype:',next(model.parameters()).dtype)
    print('Parameters:',sum(p.numel() for p in model.parameters()))


if __name__=='__main__':main()
