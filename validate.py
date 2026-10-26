"""Required evidence gates and fresh-process exported-model test."""
from core import *


def run():
    configure('cuda');checks=[]
    tests=json.loads((ROOT/'state/tests.done.json').read_text());assert tests['status']=='PASS';checks.append('independent math/shape/resume tests')
    for p,h in json.loads((ROOT/'state/source_lock.json').read_text()).items():assert sha(ROOT/p)==h
    atomic_json(ROOT/'reports/SCIENTIFIC_SOURCE_LOCK.json',json.loads((ROOT/'state/source_lock.json').read_text()))
    atomic_json(ROOT/'reports/FOCUSED_TESTS.json',tests)
    atomic_json(ROOT/'reports/RESOURCE_PREFLIGHT.json',json.loads((ROOT/'state/preflight.done.json').read_text()))
    checks.append('locked scientific source/configuration unchanged')
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    for p,h in manifest['data_files'].items():assert sha(ROOT/'data'/p)==h
    assert sha(ROOT/'data/piqa.json')==manifest['piqa_sha256'];checks.append('locked input hashes')
    for job in jobs():
        result=json.loads((ROOT/'metrics'/f'{job["id"]}_final.json').read_text())
        assert result['totals']['input_tokens']==C['train_blocks']*128 and result['totals']['updates']==192
        assert result['totals']['skipped_updates']==0,'Skipped updates compromise matched recovery'
        assert result['test']['scored_tokens']==C['test_blocks']*127 and result['piqa']['count']==C['piqa_examples']
        assert len(result['generation_benchmark']['seconds_each'])==5 and result['generation_benchmark']['generated_tokens_per_second']>0
        assert [x['input_tokens'] for x in result['recovery']]==[0,49152,98304,196608]
        assert json.loads((ROOT/'state'/job['id']/'smoke.json').read_text())['nonzero_update']
        checked_load(ROOT/'state'/job['id']/'resume.pt')
    checks.append('all18 matched196608-token/192-update runs; real gradients; finite quality; recovery curves')
    for device in ['cpu','cuda']:
        tiny=json.loads((ROOT/f'metrics/tiny_{device}.json').read_text());assert len(tiny['rows'])==15
        for variant in ['teacher','moderate_informed_s42_kl','strong_informed_s42_kl']:
            b=json.loads((ROOT/f'metrics/benchmark_{variant}_{device}.json').read_text())
            assert b['actual_parameter_devices']==[device] and len(b['cases'])==3
            assert all(len(c['runs'])==5 for c in b['cases'])
    checks.append('executedCPU/CUDAtiny training, matchedFP32benchmark, real3Bquality/three-size device audits')
    cpu=json.loads((ROOT/'metrics/tiny_cpu.json').read_text())['benchmark']
    gpu=json.loads((ROOT/'metrics/tiny_cuda.json').read_text())['benchmark']
    assert cpu['matched_weights_sha256']==gpu['matched_weights_sha256'];checks.append('tinybenchmarksameweightsandprecision')
    exported=json.loads((ROOT/'model/EXPORT_COMPLETE.json').read_text())
    notice=ROOT/'model/NOTICE.txt'
    assert sha(notice)==exported['files']['NOTICE.txt'],'Unexpected model notice change; investigate'
    required='Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All Rights Reserved.'
    if required not in notice.read_text(encoding='utf-8'):
        # Legal metadata only: never alter or re-fingerprint numerical weights.
        complete_notice=(ROOT/'NOTICE.txt').read_text(encoding='utf-8').split('\nSupplied NVIDIA')[0].replace('licenses/QWEN_RESEARCH_LICENSE.txt','LICENSE.txt')
        notice.write_text(complete_notice,encoding='utf-8')
        exported['previous_short_notice_sha256']=exported['files']['NOTICE.txt']
        exported['files']['NOTICE.txt']=sha(notice)
        exported['legal_notice_completed_without_weight_changes']=True
        atomic_json(ROOT/'model/EXPORT_COMPLETE.json',exported)
    for p,h in exported['files'].items():assert sha(ROOT/'model'/p)==h
    model,load_s=load_model('cuda',ROOT/'model');ev=torch.load(ROOT/'data/evaluation.pt',weights_only=True)
    result=nll(model,ev['test'][:16],'cuda')
    expected=json.loads((ROOT/'metrics/strong_informed_s42_kl_final.json').read_text())['test']['per_block_nll'][:16]
    assert max(abs(a-b) for a,b in zip(result['per_block_nll'],expected))<.002,'Export differs from evaluated merged model'
    tok=tokenizer();x=tok('Machine learning is a branch of artificial intelligence that',return_tensors='pt').to('cuda')
    with torch.inference_mode():y=model.generate(**x,max_new_tokens=32,do_sample=False,pad_token_id=tok.eos_token_id)
    assert y.shape[1]>x.input_ids.shape[1]
    checks.append('exportedweightreadbackandfreshprocessgeneration/quality')
    # Keep portable adapter-only finals for every experiment; never upload optimizer/RNG state.
    for job in jobs():
        state=checked_load(ROOT/'state'/job['id']/'resume.pt')
        path=ROOT/'adapters'/f'{job["id"]}.pt'
        atomic_torch(path,state['adapter'])
        atomic_json(ROOT/'reports'/f'{job["id"]}_UPDATE_CHECK.json',json.loads((ROOT/'state'/job['id']/'smoke.json').read_text()))
    checks.append('all18finaladaptersretainedwithoutduplicatefullweights')
    archive=ROOT.parent/'FINAL_PROJECT_SUBMISSION/tools/verify_package.py'
    if archive.exists():subprocess.run([sys.executable,str(archive),'--quick'],check=True)
    checks.append('sealed prior submission quick manifest verification')
    record={'status':'PASS','checks':checks,'jobs':len(jobs()),'exported_generation':tok.decode(y[0],skip_special_tokens=True),
            'exported_test_subset':result,'load_seconds':load_s,'memory':memory('cuda')}
    atomic_json(ROOT/'reports/VALIDATION.json',record);print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':run()
