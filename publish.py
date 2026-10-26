"""Publish only validated evidence; default private; never reads or prints secrets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def command(args,check=True):
    return subprocess.run(args,cwd=ROOT,text=True,encoding='utf-8',capture_output=True,check=check)


def main():
    p=argparse.ArgumentParser();p.add_argument('--public',action='store_true');p.add_argument('--name',default='minitron-pruning-under-8gb');p.add_argument('--gh',default=shutil.which('gh'))
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    assert re.fullmatch(r'[a-zA-Z0-9_.-]+',a.name),'Repository name only, no remote or shell arguments'
    assert json.loads((ROOT/'reports/VALIDATION.json').read_text())['status']=='PASS'
    assert json.loads((ROOT/'presentation/VALIDATION.json').read_text(encoding='utf-8-sig'))['status']=='PASS'
    build=json.loads((ROOT/'presentation/BUILD_SOURCES.json').read_text())
    assert hashlib.sha256((ROOT/'reports/SUMMARY.json').read_bytes()).hexdigest()==build['summary_sha256'],'Deck uses a stale measured summary'
    assert hashlib.sha256((ROOT/'presentation/slides.json').read_bytes()).hexdigest()==build['slides_sha256'],'Deck text changed after validated build'
    allowed=[]
    for pattern in ['*.py','*.md','*.ps1','config.json','requirements.txt','NOTICE.txt','.gitignore','.gitattributes','examples/*.yml','data/*','recipes/*','metrics/*.json',
                    'reports/*','figures/*','presentation/*','presentation/previews/*','licenses/*','adapters/*','logs/stage_costs.jsonl','logs/*_s*.jsonl']:
        for path in ROOT.glob(pattern):
            if not path.is_file() or path.name.endswith('.tmp') or path.name.endswith('.partial.json'):continue
            if '.local.' in path.name:continue
            if path.suffix.lower() in ['.safetensors','.npy']:continue
            if path.suffix.lower()=='.pdf' and path.parent not in [ROOT/'presentation',ROOT/'reports']:continue
            if path.name in ['CURRENT_RUN_CHECKPOINT.md','PUBLICATION_INVENTORY.json']:continue
            allowed.append(path)
    allowed=sorted(set(allowed));total=0
    for path in allowed:
        assert path.stat().st_size<90*2**20,'Oversized GitHub file: '+path.name
        total+=path.stat().st_size
        if path.suffix in ['.py','.json','.md','.csv','.ps1','.jsonl','.txt']:
            text=path.read_text(encoding='utf-8-sig')
            assert not re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{24,}',text),'Possible secret; not uploading'
            # Reproduction documentation may mention the project root, not private user homes.
            assert not re.search(r'[A-Za-z]:[\\/]+Users[\\/]+',text,re.I),'Private user path in '+str(path.relative_to(ROOT))
    assert total<300*2**20,'Repository unexpectedly large; inspect rather than upload'
    inventory={'files':[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in allowed],
               'total_bytes':total,'excluded':['large model weights','optimizer/RNG states','teacher target payloads','environment','private run logs','supplied copyrighted PDFs']}
    (ROOT/'reports/PUBLICATION_INVENTORY.json').write_text(json.dumps(inventory,indent=2))
    if a.prepare_only:
        print('Publication inventory prepared; no remote created or data uploaded.');return
    if not a.gh:raise RuntimeError('GitHub CLI unavailable; pass --gh path/to/gh.exe')
    auth=command([a.gh,'auth','status'],check=False)
    if auth.returncode:raise RuntimeError('GitHub is not authenticated. Run gh auth login --web; never paste credentials into reports/chat.')
    user=json.loads(command([a.gh,'api','user']).stdout);login=user['login']
    if not (ROOT/'.git').exists():command(['git','init','-b','main'])
    command(['git','config','pack.threads','2'])
    # Repo-local metadata only; do not change user-wide git settings.
    if command(['git','config','user.name'],check=False).returncode:command(['git','config','user.name',user.get('name') or login])
    if command(['git','config','user.email'],check=False).returncode:command(['git','config','user.email',f'{user["id"]}+{login}@users.noreply.github.com'])
    for path in allowed+[ROOT/'reports/PUBLICATION_INVENTORY.json']:
        command(['git','add','--',path.relative_to(ROOT).as_posix()])
    staged=command(['git','diff','--cached','--name-only']).stdout.splitlines()
    if staged:command(['git','commit','-m','Publish validated pruning, distillation and CPU GPU study'])
    remote=command(['git','remote','get-url','origin'],check=False)
    if remote.returncode:
        command([a.gh,'repo','create',a.name,'--public' if a.public else '--private','--source',str(ROOT),'--remote','origin',
                 '--description','Minitron-inspired Qwen structured pruning, full-vocabulary KD and measured CPU/CUDA protocol under 8 GiB VRAM'])
    else:
        assert remote.stdout.strip() in [f'https://github.com/{login}/{a.name}.git',f'https://github.com/{login}/{a.name}'], 'Unexpected existing remote protected'
    # Explicit gh-authenticated helper; does not expose tokens or alter global credentials.
    helper='!'+str(Path(a.gh).as_posix())+' auth git-credential'
    command(['git','-c','credential.helper=','-c','credential.helper='+helper,'push','-u','origin','main'])
    view=json.loads(command([a.gh,'repo','view',f'{login}/{a.name}','--json','url,visibility,defaultBranchRef']).stdout)
    local=command(['git','rev-parse','HEAD']).stdout.strip()
    # Verify the hosted branch through GitHub, not only a local origin name.
    hosted=command([a.gh,'api',f'repos/{login}/{a.name}/commits/main','--jq','.sha']).stdout.strip()
    assert local==hosted,'Remote commit verification failed'
    receipt={'status':'PUBLISHED_AND_VERIFIED','url':view['url'],'visibility':view['visibility'],'commit':local,'files':len(allowed),'bytes':total}
    (ROOT/'publish_receipt.local.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
