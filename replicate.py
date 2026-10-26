"""Create an independent run from published locked inputs; preserve old results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--teacher',type=Path,required=True,help='Local pinned Qwen2.5-3B snapshot, not an API/model ID')
    p.add_argument('--run',action='store_true',help='Explicitly start the hours-long independent replication after preparation')
    a=p.parse_args();out=a.output.resolve();teacher=a.teacher.resolve();start=time.perf_counter()
    if out.exists():raise RuntimeError('Existing output protected: choose a NEW directory')
    if not (teacher/'config.json').is_file() or not list(teacher.glob('*.safetensors')):raise RuntimeError('A local complete teacher checkpoint is required')
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    for name,h in manifest['data_files'].items():assert digest(ROOT/'data'/name)==h,'Prepared tensor hash mismatch'
    assert digest(ROOT/'data/piqa.json')==manifest['piqa_sha256']
    fingerprint=ROOT/'reports/BASE_MODEL_FINGERPRINT.json'
    if fingerprint.exists():
        for name,h in json.loads(fingerprint.read_text())['files'].items():assert digest(teacher/name)==h,'Teacher differs from measured pinned base: '+name
    out.mkdir(parents=True)
    for path in ROOT.iterdir():
        if path.is_file() and path.suffix in ['.py','.md','.txt','.ps1'] and '.local.' not in path.name:shutil.copy2(path,out/path.name)
    for name in ['config.json','.gitignore']:shutil.copy2(ROOT/name,out/name)
    for name in ['data','licenses']:shutil.copytree(ROOT/name,out/name)
    (out/'presentation').mkdir();shutil.copy2(ROOT/'presentation/render.ps1',out/'presentation/render.ps1')
    (out/'reports').mkdir()
    for name in ['BACKEND_SUPPORT_AUDIT.md','PROTOCOL_AMENDMENTS.md','BASE_MODEL_FINGERPRINT.json']:
        if (ROOT/'reports'/name).exists():shutil.copy2(ROOT/'reports'/name,out/'reports'/name)
    for name in ['state','logs','cache','metrics','recipes','figures','adapters']:(out/name).mkdir()
    shutil.copy2(ROOT/'adapters/NOTICE.txt',out/'adapters/NOTICE.txt')
    manifest['historical_original_prepare_seconds']=manifest['prepare_seconds']
    manifest['prepare_seconds']=time.perf_counter()-start
    manifest['preparation_mode']='Reused byte-identical published prepared inputs; no legacy workspace required'
    (out/'data/manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    env=dict(os.environ,LEVEL4_TEACHER=str(teacher))
    bootstrap="from core import *; configure('cpu'); atomic_json(ROOT/'environment.json', environment()); atomic_json(ROOT/'state/prepare.done.json', {'status':'PASS_REUSED_VERIFIED_INPUTS','no_training_or_evaluation_claim':True})"
    subprocess.run([sys.executable,'-c',bootstrap],cwd=out,env=env,check=True)
    # Machine-local start script; excluded from any later publication.
    q=lambda value:str(value).replace("'","''")
    launcher=f"$env:LEVEL4_TEACHER = '{q(teacher)}'\n& '{q(sys.executable)}' -u (Join-Path $PSScriptRoot 'run.py')\n"
    (out/'START_REPLICA.local.ps1').write_text(launcher,encoding='utf-8')
    print('Prepared independent run at',out,'without copying old metrics, checkpoints or final models.')
    if a.run:subprocess.run([sys.executable,'-u',str(out/'run.py')],cwd=out,env=env,check=True)
    else:print('No training started. Use START_REPLICA.local.ps1 there when ready, or pass --run explicitly.')


if __name__=='__main__':main()
