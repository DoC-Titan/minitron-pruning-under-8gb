"""Durable sequential experiment driver. Rerun after power loss; no paid services."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import msvcrt
from core import *


def call(script,*args):
    guard('cpu',check_size=True)
    if (ROOT/'STOP').exists():raise RuntimeError('STOP requested: remove STOP to continue')
    label=script+' '+' '.join(str(x) for x in args)
    atomic_json(ROOT/'RUN_STATUS.json',{'status':'RUNNING','stage':label,'updated':dt.datetime.now().astimezone().isoformat(),
               'resume':'Run RESUME.ps1 after restoring power. No automatic startup task was installed.'})
    print('\nSTAGE',label,flush=True);start=time.perf_counter()
    with (ROOT/'logs/runner.log').open('a',encoding='utf-8') as f:
        f.write('\nSTAGE '+label+'\n');f.flush()
        result=subprocess.run([sys.executable,'-u',str(ROOT/script),*[str(x) for x in args]],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    append(ROOT/'logs/stage_costs.jsonl',{'stage':label,'seconds':time.perf_counter()-start,'returncode':result.returncode,
                                        'finished':dt.datetime.now().astimezone().isoformat()})
    if result.returncode:raise RuntimeError('Stage failed safely: '+label+'; inspect logs/runner.log')


def verify_lock():
    paths=['config.json','core.py','worker.py','tests.py','tiny_reference.py','run.py']
    current={p:sha(ROOT/p) for p in paths}
    path=ROOT/'state/source_lock.json'
    if path.exists():assert json.loads(path.read_text())==current,'Run source/config changed; do not mix protocols'
    else:atomic_json(path,current)


def run():
    for name in ['logs','state','metrics','cache']:(ROOT/name).mkdir(exist_ok=True)
    verify_lock()
    if not (ROOT/'state/prepare.done.json').exists():call('worker.py','prepare','--device','cpu')
    if not (ROOT/'state/tests.done.json').exists():call('tests.py')
    assert json.loads((ROOT/'state/tests.done.json').read_text())['status']=='PASS'
    if not (ROOT/'state/calibrate.done.json').exists():call('worker.py','calibrate')
    if not (ROOT/'state/preflight.done.json').exists():call('worker.py','preflight')
    if not (ROOT/'metrics/teacher_final.json').exists():call('worker.py','teacher')
    # CPU and CUDA phases serialized, not competing for benchmark resources.
    for device in ['cpu','cuda']:
        if not (ROOT/f'metrics/tiny_{device}.json').exists():call('tiny_reference.py','--device',device)
    for shard in range(math.ceil(C['train_blocks']/C['shard_blocks'])):
        done=ROOT/'state'/f'shard_{shard:03d}.complete.json'
        if done.exists():continue
        consumers=[j for j in jobs() if not (ROOT/'state'/j['id']/f'shard_{shard:03d}.done.json').exists()]
        if any(j['objective']=='kl' for j in consumers):call('worker.py','cache','--shard',shard)
        for job in consumers:call('worker.py','train','--job',job['id'],'--shard',shard)
        # Every consumer receipt validated before reclaiming OUR generated cache only.
        for job in jobs():
            r=json.loads((ROOT/'state'/job['id']/f'shard_{shard:03d}.done.json').read_text());assert r['next_shard']>shard
        target=ROOT/'cache'/f'full_logits_{shard:03d}.npy'
        if target.exists():
            receipt=json.loads(target.with_suffix('.json').read_text());assert sha(target)==receipt['sha256']
            target.unlink()  # bounded, explicitly owned regenerable cache
        atomic_json(done,{'status':'PASS','all_consumers':len(jobs()),'target_payload_reclaimed':True})
    for job in jobs():
        if not (ROOT/'metrics'/f'{job["id"]}_final.json').exists():call('worker.py','evaluate','--job',job['id'])
    variants=['teacher','moderate_informed_s42_kl','strong_informed_s42_kl']
    for variant in variants:
        for device in ['cpu','cuda']:
            if not (ROOT/'metrics'/f'benchmark_{variant}_{device}.json').exists():call('worker.py','benchmark','--job',variant,'--device',device)
    if not (ROOT/'model/EXPORT_COMPLETE.json').exists():call('worker.py','export','--job','strong_informed_s42_kl')
    # Validated reload from exported checkpoint, not merely in-memory construction.
    call('validate.py')
    call('analyze.py')
    call('build_presentation.py')
    atomic_json(ROOT/'RUN_STATUS.json',{'status':'EXPERIMENTS_AND_ARTIFACTS_COMPLETE','updated':dt.datetime.now().astimezone().isoformat(),
        'next':'GitHub publication requires authenticated account; see publish.py. Never claim publication without receipt.'})


if __name__=='__main__':
    (ROOT/'state').mkdir(exist_ok=True)
    lock=(ROOT/'state/runner.lock').open('a+b');lock.seek(0);lock.write(b'1');lock.flush();lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise SystemExit('Another runner holds this project lock; not launching a second GPU job.')
    try:run()
    except Exception:
        atomic_json(ROOT/'RUN_STATUS.json',{'status':'PAUSED_OR_FAILED','error':traceback.format_exc(),'updated':dt.datetime.now().astimezone().isoformat(),
                                           'resume':'Inspect logs, remove STOP if requested, then run RESUME.ps1. Existing valid checkpoints are preserved.'})
        raise
    finally:lock.close()
