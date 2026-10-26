"""One bounded stage per process. Never loads teacher and student concurrently."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import time
import traceback

from core import *


def data(name): return torch.load(ROOT / 'data' / (name + '.pt'), weights_only=True)


def prepare():
    start = time.perf_counter(); workspace = ROOT.parent
    for folder in ['data', 'metrics', 'logs', 'state', 'recipes', 'cache', 'figures', 'reports', 'presentation']:
        (ROOT/folder).mkdir(exist_ok=True)
    atomic_json(ROOT/'environment.json', environment())
    source = workspace/'upgrade_v2/data'
    train = torch.load(source/'train.pt', weights_only=True)[:C['train_blocks']].clone()
    cal = torch.load(source/'calibration.pt', weights_only=True)[:C['calibration_blocks']].clone()
    ev = torch.load(source/'evaluation.pt', weights_only=True)
    ev = {'validation': ev['validation'][:C['validation_blocks']].clone(), 'test': ev['test'][:C['test_blocks']].clone()}
    pools = [set(tensor_hash(row) for row in group) for group in [train, cal, ev['validation'], ev['test']]]
    assert all(pools[i].isdisjoint(pools[j]) for i in range(4) for j in range(i))
    for name, value in [('train', train), ('calibration', cal), ('evaluation', ev)]:
        atomic_torch(ROOT/'data'/f'{name}.pt', value)
    task = json.loads((source/'piqa.json').read_text())
    task['records'] = task['records'][:C['piqa_examples']]
    atomic_json(ROOT/'data/piqa.json', task)
    # Byte LM reference uses official split boundaries; tokenizer fixed a priori.
    from datasets import Dataset
    byte_data = {}
    for split, length in [('train', 262144), ('validation', 32768), ('test', 32768)]:
        arrow = next((workspace/'hf-cache/datasets').rglob(f'wikitext-{split}.arrow'))
        ds = Dataset.from_file(str(arrow))
        raw = '\n\n'.join(ds['text']).encode('utf-8')[:length]
        byte_data[split] = torch.tensor(list(raw), dtype=torch.long)
    atomic_torch(ROOT/'data/byte_corpus.pt', byte_data)
    tok = tokenizer()
    text = tok.decode(ev['validation'].reshape(-1)[:1024])
    atomic_json(ROOT/'data/benchmark_prompt.json', {'text': text, 'source': 'validation-only, fixed before benchmarks'})
    manifest = {'train_sha256': tensor_hash(train), 'train_blocks': len(train), 'input_tokens': train.numel(),
                'causal_targets': len(train)*127, 'block_overlap': False, 'calibration_blocks': len(cal),
                'test_blocks': len(ev['test']), 'validation_blocks': len(ev['validation']),
                'source_manifest': json.loads((source/'manifest.json').read_text()),
                'data_files': {p.name: sha(p) for p in (ROOT/'data').glob('*.pt')},
                'piqa_sha256': sha(ROOT/'data/piqa.json'), 'prepare_seconds': time.perf_counter()-start,
                'byte_reference': 'UTF-8 bytes, vocab256, official train/validation/test splits; not comparable to subword Qwen PPL',
                'test_caveat': 'Reuses disclosed prior evaluation partition; no new unseen benchmark claim'}
    atomic_json(ROOT/'data/manifest.json', manifest)
    atomic_json(ROOT/'state/prepare.done.json', {'status': 'PASS', 'seconds': time.perf_counter()-start})


def calibrate():
    model, load_s = load_model(); cal = data('calibration'); start = time.perf_counter()
    baseline = nll(model, cal, 'cuda'); original = list(model.model.layers); n = len(original)
    ranks = {}
    for level, spec in C['structures'].items():
        k = n-spec['layers']; values = []
        for window in range(1, n-k):
            model.model.layers = nn.ModuleList(original[:window]+original[window+k:]); fix_layers(model)
            result = nll(model, cal, 'cuda')
            values.append({'start': window, 'removed': list(range(window, window+k)), 'nll': result['nll'], 'delta_nll': result['nll']-baseline['nll']})
            if window % 8 == 0: print('SENSITIVITY', level, window, flush=True)
            # Every window is restartable, without trusting half-written JSON.
            atomic_json(ROOT/'metrics'/f'sensitivity_{level}.partial.json', values)
        ranks[level] = values
    model.model.layers = nn.ModuleList(original); fix_layers(model)
    scores = []
    for layer in model.model.layers:
        m = layer.mlp
        scores.append((m.gate_proj.weight.float().square().sum(1)+m.up_proj.weight.float().square().sum(1)+m.down_proj.weight.float().square().sum(0)).sqrt().cpu())
    scores = torch.stack(scores); atomic_torch(ROOT/'recipes/ffn_magnitude_scores.pt', scores)
    for level, spec in C['structures'].items():
        for criterion in C['criteria']:
            for seed in C['seeds']:
                choice = random.Random(seed).choice(ranks[level]) if criterion == 'random' else min(ranks[level], key=lambda r: (r['nll'], r['start']))
                kept = [i for i in range(n) if i not in choice['removed']]
                channels = []
                for i in kept:
                    ids = (torch.randperm(model.config.intermediate_size, generator=torch.Generator().manual_seed(seed+i))[:spec['ffn']].sort().values
                           if criterion == 'random' else torch.argsort(scores[i], descending=True, stable=True)[:spec['ffn']].sort().values)
                    channels.append({'original_layer': i, 'retained': ids.tolist()})
                recipe = {'level': level, 'criterion': criterion, 'seed': seed, 'teacher_revision': C['teacher_revision'],
                          'retained_layers': kept, 'removed_layers': choice['removed'], 'ffn': spec['ffn'], 'channels': channels,
                          'selection': 'random internal contiguous window + random channels' if criterion == 'random' else '32-block contiguous-window NLL + coupled FFN L2 magnitude',
                          'removed_channels_definition': 'For each retained layer, complement of retained in range(11008); layers removed entirely have all channels removed',
                          'parameter_reduction_is_physical': True}
                atomic_json(recipe_path({'level': level, 'criterion': criterion, 'seed': seed}), recipe)
    atomic_json(ROOT/'metrics/sensitivity.json', {'baseline': baseline, 'rankings': ranks, 'seconds': time.perf_counter()-start, 'load_seconds': load_s})
    atomic_json(ROOT/'state/calibrate.done.json', {'status': 'PASS', 'seconds': time.perf_counter()-start, 'memory': memory('cuda')})


def cache(shard):
    path = ROOT/'cache'/f'full_logits_{shard:03d}.npy'
    receipt = path.with_suffix('.json')
    if receipt.exists() and path.exists():
        r = json.loads(receipt.read_text())
        if sha(path) == r['sha256']: return
        raise RuntimeError('Existing target shard failed hash; investigate instead of overwriting')
    train = data('train'); start_idx = shard*C['shard_blocks']; end = min(len(train), start_idx+C['shard_blocks'])
    assert start_idx < end
    model, load_s = load_model(); tmp = path.with_suffix('.partial.npy')
    shape = (end-start_idx, C['sequence_length']-1, model.config.vocab_size)
    begin = time.perf_counter()
    mm = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float16, shape=shape)
    with torch.inference_mode():
        for i in range(start_idx, end):
            if (i-start_idx) % 16 == 0: guard('cuda', check_size=True)
            z = model(input_ids=train[i].unsqueeze(0).cuda(), use_cache=False).logits[0, :-1]
            assert torch.isfinite(z).all(); mm[i-start_idx] = z.cpu().numpy()
    mm.flush(); del mm
    # Windows FlushFileBuffers requires a write-capable handle.
    with tmp.open('r+b') as f: os.fsync(f.fileno())
    os.replace(tmp, path)
    atomic_json(receipt, {'sha256': sha(path), 'shape': list(shape), 'dtype': 'float16', 'full_vocabulary': True,
            'train_sha256': tensor_hash(train), 'teacher_revision': C['teacher_revision'], 'start': start_idx, 'end': end,
            'inference_transfer_write_seconds': time.perf_counter()-begin, 'load_seconds': load_s,
            'actual_bytes': path.stat().st_size, 'memory': memory('cuda')})
    print('CACHED', shard, shape, flush=True)


def construct(job, device='cuda', adapters=True):
    model, load_s = load_model(device); recipe = json.loads(recipe_path(job).read_text()); start = time.perf_counter()
    apply_recipe(model, recipe); sync(device); prune_s = time.perf_counter()-start
    seed_all(job['seed'])
    if adapters: add_adapters(model)
    return model, {'load_seconds': load_s, 'physical_pruning_seconds': prune_s}


def checkpoint_job(job, model, optimizer, scaler, next_shard, cursor, totals, recovery):
    state = {'adapter': adapter_state(model), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
             'next_shard': next_shard, 'cursor': cursor, 'totals': totals, 'recovery': recovery,
             'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
             'job': job, 'config_sha256': sha(ROOT/'config.json'), 'recipe_sha256': sha(recipe_path(job))}
    atomic_torch(ROOT/'state'/job['id']/'resume.pt', state, backup=True)


def train_shard(job, shard):
    output = ROOT/'state'/job['id']; state_path = output/'resume.pt'
    state = checked_load(state_path, backup=True) if state_path.exists() or state_path.with_suffix('.previous.pt').exists() else None
    if state:
        assert state['job'] == job and state['config_sha256'] == sha(ROOT/'config.json')
        assert state['recipe_sha256'] == sha(recipe_path(job))
        if state['next_shard'] > shard:
            # Power can fail AFTER the final update commits but BEFORE its small
            # completion receipt is written. Repair from verified durable state.
            atomic_json(output/f'shard_{shard:03d}.done.json', {'status':'PASS',
                'next_shard':state['next_shard'],'totals':state['totals'],
                'recovery':state['recovery'],'checkpoint_sha256':sha(state_path if state_path.exists() else state_path.with_suffix('.previous.pt')),
                'receipt_recovered_from_checkpoint':True})
            return
        assert state['next_shard'] == shard
    else: assert shard == 0
    model, construction = construct(job); model.config.use_cache = False
    model.enable_input_require_grads(); model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=C['lr'], weight_decay=0)
    scaler = torch.amp.GradScaler('cuda', init_scale=128)
    totals = {'model_execution_seconds': 0., 'target_read_transfer_seconds': 0., 'validation_seconds': 0.,
              'checkpoint_seconds': 0., 'construction_seconds': 0., 'input_tokens': 0, 'updates': 0,
              'training_peak_allocated_gib': 0., 'losses': [], 'skipped_updates': 0}
    recovery = []
    if state:
        model.load_state_dict(state['adapter'], strict=False); opt.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
        totals = state['totals']; recovery = state['recovery']; cursor = state['cursor']
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state(state['cuda_rng']); del state
    else: cursor = 0
    totals['construction_seconds'] += sum(construction.values())
    if shard == 0 and cursor == 0:
        result = nll(model, data('evaluation')['validation'], 'cuda')
        totals['validation_seconds'] += result['seconds']; recovery.append({'input_tokens': 0, 'validation': result})
        atomic_json(ROOT/'metrics'/f'{job["id"]}_initial.json', {'job': job,
                    'parameters': sum(p.numel() for p in model.parameters())-sum(p.numel() for p in params),
                    'trainable_parameters': sum(p.numel() for p in params), 'validation': result,
                    'construction': construction, 'recipe_sha256': sha(recipe_path(job))})
        checkpoint_job(job, model, opt, scaler, shard, cursor, totals, recovery)
    train = data('train'); begin = shard*C['shard_blocks']; count = min(C['shard_blocks'], len(train)-begin)
    order = torch.randperm(count, generator=torch.Generator().manual_seed(job['seed']+100000*shard)).tolist()
    target_map = None
    if job['objective'] == 'kl':
        target_path = ROOT/'cache'/f'full_logits_{shard:03d}.npy'
        receipt = json.loads(target_path.with_suffix('.json').read_text())
        assert receipt['train_sha256'] == tensor_hash(train) and receipt['teacher_revision'] == C['teacher_revision']
        target_map = np.load(target_path, mmap_mode='r')
    assert cursor % C['accumulation'] == 0
    model.train(); opt.zero_grad(set_to_none=True)
    for offset in range(cursor, count, C['accumulation']):
        sample = guard('cuda'); torch.cuda.reset_peak_memory_stats(); losses = []; execution = 0.; transfers = 0.
        before_update = adapter_state(model) if totals['updates'] == 0 else None
        for j in range(offset, offset+C['accumulation']):
            idx = order[j]; sync('cuda'); t = time.perf_counter()
            x = train[begin+idx].unsqueeze(0).cuda()
            target = torch.from_numpy(np.array(target_map[idx], copy=True)).cuda() if target_map is not None else None
            sync('cuda'); transfers += time.perf_counter()-t; t = time.perf_counter()
            logits = model(input_ids=x, use_cache=False).logits[0, :-1]
            loss = full_kl(logits, target) if target is not None else F.cross_entropy(logits.float(), x[0, 1:])
            assert torch.isfinite(loss), 'Nonfinite loss'
            scaler.scale(loss/C['accumulation']).backward(); losses.append(float(loss))
            sync('cuda'); execution += time.perf_counter()-t
            del logits, loss, x, target
        t = time.perf_counter(); scaler.unscale_(opt); norm = torch.nn.utils.clip_grad_norm_(params, 1.)
        assert torch.isfinite(norm), 'Nonfinite gradients'
        scale_before = scaler.get_scale(); scaler.step(opt); scaler.update(); sync('cuda')
        if scaler.get_scale() < scale_before: totals['skipped_updates'] += 1
        opt.zero_grad(set_to_none=True); execution += time.perf_counter()-t
        if before_update is not None:
            changed = any(not torch.equal(v, before_update[k]) for k,v in adapter_state(model).items())
            assert changed and float(norm)>0, 'No real optimizer update'
            atomic_json(output/'smoke.json', {'nonzero_update': True, 'gradient_norm': float(norm), 'loss': statistics.mean(losses),
                                             'memory': memory('cuda'), 'smoke_included_as_first_planned_update': True})
        totals['updates'] += 1; totals['input_tokens'] += C['accumulation']*128
        totals['model_execution_seconds'] += execution; totals['target_read_transfer_seconds'] += transfers
        totals['training_peak_allocated_gib'] = max(totals['training_peak_allocated_gib'], torch.cuda.max_memory_allocated()/GIB)
        totals['losses'].append(statistics.mean(losses))
        done_blocks = (shard*C['shard_blocks']+offset+C['accumulation'])
        if done_blocks in C['recovery_blocks']:
            val = nll(model, data('evaluation')['validation'], 'cuda'); model.train()
            totals['validation_seconds'] += val['seconds']; recovery.append({'input_tokens': done_blocks*128, 'validation': val})
            atomic_torch(output/f'adapter_{done_blocks*128}.pt', adapter_state(model))
        next_shard = shard+1 if offset+C['accumulation'] == count else shard
        next_cursor = 0 if next_shard > shard else offset+C['accumulation']
        t = time.perf_counter(); checkpoint_job(job, model, opt, scaler, next_shard, next_cursor, totals, recovery)
        totals['checkpoint_seconds'] += time.perf_counter()-t
        append(ROOT/'logs'/f'{job["id"]}.jsonl', {'update': totals['updates'], 'input_tokens': totals['input_tokens'],
                 'loss': statistics.mean(losses), 'gradient_norm': float(norm), 'execution_seconds': execution,
                 'transfer_seconds': transfers, 'gpu_sample': sample})
        if (ROOT/'STOP').exists(): raise RuntimeError('USER PAUSE: durable checkpoint saved. Remove STOP and run RESUME.ps1')
    atomic_json(output/f'shard_{shard:03d}.done.json', {'status': 'PASS', 'next_shard': shard+1, 'totals': totals,
                 'recovery': recovery, 'checkpoint_sha256': sha(state_path)})
    print('TRAINED', job['id'], shard, totals['input_tokens'], flush=True)


def merged_job(job, device='cuda'):
    model, times = construct(job, device, adapters=False)
    state = checked_load(ROOT/'state'/job['id']/'resume.pt', backup=True)
    assert state['totals']['input_tokens'] == C['train_blocks']*128
    start = time.perf_counter()
    with torch.no_grad():
        for i, layer in enumerate(model.model.layers):
            for key in ['q_proj', 'v_proj']:
                prefix = f'model.layers.{i}.self_attn.{key}'
                delta = state['adapter'][prefix+'.B'] @ state['adapter'][prefix+'.A'] * C['alpha']/C['rank']
                # Define one FP16 merged checkpoint for CPU/GPU scientific comparison.
                weight = getattr(layer.self_attn, key).weight
                merged = (weight.detach().half().cpu() + delta.half()).half()
                weight.copy_(merged.to(weight.device, dtype=weight.dtype))
    times['merge_seconds'] = time.perf_counter()-start
    return model, state, times


def final_eval(job):
    path = ROOT/'metrics'/f'{job["id"]}_final.json'
    if path.exists(): return
    model, state, times = merged_job(job); start = time.perf_counter(); ev = data('evaluation')
    result = {'job': job, 'parameters': sum(p.numel() for p in model.parameters()),
              'test': nll(model, ev['test'], 'cuda'), 'validation': nll(model, ev['validation'], 'cuda'),
              'piqa': piqa(model, tokenizer(), json.loads((ROOT/'data/piqa.json').read_text())['records'], 'cuda'),
              'totals': state['totals'], 'recovery': state['recovery'], 'construction': times,
              'device': 'cuda', 'dtype': 'float16', 'merge': 'FP16 base + FP16-rounded FP32 LoRA delta; no live adapter overhead',
              'memory': memory('cuda'), 'evaluation_seconds': time.perf_counter()-start}
    # Measure every recovered model; do not infer random-student throughput from
    # an informed student merely because their tensor shapes match.
    prompt=ev['validation'][0].unsqueeze(0).cuda();durations=[]
    settings=dict(input_ids=prompt,attention_mask=torch.ones_like(prompt),min_new_tokens=16,max_new_tokens=16,
                  do_sample=False,use_cache=True,pad_token_id=model.config.eos_token_id)
    with torch.inference_mode():
        for _ in range(2):model.generate(**settings)
        sync('cuda');torch.cuda.reset_peak_memory_stats()
        for _ in range(5):
            sync('cuda');t=time.perf_counter();out=model.generate(**settings);sync('cuda')
            assert out.shape[-1]-prompt.shape[-1]==16;durations.append(time.perf_counter()-t)
    result['generation_benchmark']={'prompt_tokens':128,'output_tokens':16,'warmups':2,'timing_repeats':5,
              'seconds_each':durations,'median_seconds':statistics.median(durations),
              'generated_tokens_per_second':16/statistics.median(durations),'memory':memory('cuda'),
              'scope':'FP16CUDA,greedy,batch1,prefill+decode; excludes reconstruction/loading and tokenization'}
    result['quality_evaluation_seconds']=result['evaluation_seconds']
    result['evaluation_seconds']=time.perf_counter()-start
    result['parameter_reduction_percent'] = 100*(1-result['parameters']/3085938688)
    atomic_json(path, result); print('FINAL', job['id'], result['test']['perplexity'], result['piqa']['accuracy'], flush=True)


def preflight():
    model, load_s = load_model(); train = data('train')
    with torch.inference_mode(): target = model(input_ids=train[0].unsqueeze(0).cuda(), use_cache=False).logits[0, :-1].cpu()
    # More memory-demanding moderate structure, same actual training primitive.
    job = next(j for j in jobs() if j['level']=='moderate' and j['criterion']=='informed' and j['objective']=='kl')
    apply_recipe(model, json.loads(recipe_path(job).read_text())); seed_all(job['seed']); add_adapters(model)
    model.enable_input_require_grads(); model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache=False; model.train(); params=[p for p in model.parameters() if p.requires_grad]
    initial=adapter_state(model); opt=torch.optim.AdamW(params,lr=C['lr']); scaler=torch.amp.GradScaler('cuda',init_scale=128)
    torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
    for _ in range(2):
        z=model(input_ids=train[0].unsqueeze(0).cuda(),use_cache=False).logits[0,:-1]
        loss=full_kl(z,target.cuda()); scaler.scale(loss/2).backward()
        del z
    scaler.unscale_(opt); norm=torch.nn.utils.clip_grad_norm_(params,1.); assert torch.isfinite(norm) and float(norm)>0
    scaler.step(opt);scaler.update()
    assert any(not torch.equal(v,initial[k]) for k,v in adapter_state(model).items())
    result={'status':'PASS','seconds':time.perf_counter()-start,'load_seconds':load_s,'loss':float(loss),'gradient_norm':float(norm),
            'memory':memory('cuda'),'nonzero_update':True,'training_state_discarded':True,'note':'Smoke tokens excluded; all real jobs start fresh'}
    atomic_json(ROOT/'state/preflight.done.json',result); print(json.dumps(result,indent=2),flush=True)


def teacher_eval():
    model, load_s = load_model(); ev=data('evaluation'); start=time.perf_counter()
    r={'job':{'id':'teacher'},'parameters':sum(p.numel() for p in model.parameters()),'load_seconds':load_s,
       'validation':nll(model,ev['validation'],'cuda'),'test':nll(model,ev['test'],'cuda'),
       'piqa':piqa(model,tokenizer(),json.loads((ROOT/'data/piqa.json').read_text())['records'],'cuda'),
       'memory':memory('cuda'),'dtype':'float16','device':'cuda','evaluation_seconds':time.perf_counter()-start}
    atomic_json(ROOT/'metrics/teacher_final.json',r)


@torch.inference_mode()
def benchmark(variant, device):
    output=ROOT/'metrics'/f'benchmark_{variant}_{device}.json'
    if output.exists():return
    if variant=='teacher':
        model,load_s=load_model(device,TEACHER); construction={'load_seconds':load_s}
    else:
        job=next(j for j in jobs() if j['id']==variant)
        model,_,construction=merged_job(job,device); load_s=sum(construction.values())
    tok=tokenizer(); ev=data('evaluation'); start=time.perf_counter()
    quality={'test':nll(model,ev['test'][:C['cpu_quality_blocks']],device),
             'piqa':piqa(model,tok,json.loads((ROOT/'data/piqa.json').read_text())['records'][:C['cpu_piqa_examples']],device)}
    text=json.loads((ROOT/'data/benchmark_prompt.json').read_text())['text']; cases=[]
    for length in C['benchmark_prompt_lengths']:
        def request():
            t=time.perf_counter(); encoded=tok(text,return_tensors='pt',add_special_tokens=False,truncation=True,max_length=length)
            assert encoded.input_ids.shape[1]==length
            preprocess=time.perf_counter()-t; t=time.perf_counter(); inputs=encoded.to(device); sync(device)
            transfer=time.perf_counter()-t; t=time.perf_counter()
            out=model.generate(**inputs,min_new_tokens=C['benchmark_new_tokens'],max_new_tokens=C['benchmark_new_tokens'],
                               do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id)
            sync(device); execution=time.perf_counter()-t; t=time.perf_counter()
            continuation=tok.decode(out[0,length:].cpu(),skip_special_tokens=True); sync(device)
            postprocess=time.perf_counter()-t
            return {'preprocess_seconds':preprocess,'transfer_seconds':transfer,'execution_seconds':execution,
                    'postprocess_seconds':postprocess,'request_e2e_seconds':preprocess+transfer+execution+postprocess,
                    'continuation':continuation}
        guard(device)
        for _ in range(C['benchmark_warmups']):request()
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        records=[request() for _ in range(C['benchmark_repeats'])]
        med={k:statistics.median(r[k] for r in records) for k in records[0] if k.endswith('_seconds')}
        durations=[r['execution_seconds'] for r in records]; q=statistics.quantiles(durations,n=4)
        cases.append({'prompt_tokens':length,'new_tokens':C['benchmark_new_tokens'],'batch_size':1,'runs':records,
                      'median':med,'execution_iqr_seconds':q[2]-q[0],'execution_range_seconds':[min(durations),max(durations)],
                      'generated_tokens_per_second':C['benchmark_new_tokens']/med['execution_seconds'],
                      'load_inclusive_first_request_seconds':load_s+records[0]['request_e2e_seconds'],'memory':memory(device)})
        atomic_json(output.with_suffix('.partial.json'),{'cases':cases,'quality':quality,'load_seconds':load_s})
    result={'variant':variant,'device':device,'dtype':str(next(model.parameters()).dtype),'actual_parameter_devices':sorted(set(p.device.type for p in model.parameters())),
            'parameters':sum(p.numel() for p in model.parameters()),'model_load_seconds':load_s,'construction_phases':construction,'cases':cases,'quality':quality,
            'cpu_threads':torch.get_num_threads(),'warmups':C['benchmark_warmups'],'repeats':C['benchmark_repeats'],
            'comparison_label':'CPU FP32 vs CUDA FP16 is deployment device+precision, not isolated hardware speedup',
            'loading_caveat':'Fresh process/model load; OS file cache is uncontrolled, not cold storage latency',
            'phase_seconds':time.perf_counter()-start}
    atomic_json(output,result)


def export(variant):
    job=next(j for j in jobs() if j['id']==variant); out=ROOT/'model'
    if (out/'EXPORT_COMPLETE.json').exists():return
    assert not out.exists() or not list(out.glob('*.safetensors')), 'Partial export exists: investigate before replacement'
    model,state,times=merged_job(job); start=time.perf_counter()
    model.save_pretrained(out,safe_serialization=True,max_shard_size='2GB');tokenizer().save_pretrained(out)
    license_path=ROOT.parent/'pilot/licenses/QWEN_RESEARCH_LICENSE.txt'
    shutil.copy2(license_path,out/'LICENSE.txt')
    (out/'NOTICE.txt').write_text('Built with Qwen. Noncommercial Qwen Research License. Modified: physical layer/FFN pruning and merged LoRA recovery. See recipe and model card.\n')
    shutil.copy2(recipe_path(job),out/'pruning_recipe.json')
    atomic_json(out/'EXPORT_COMPLETE.json',{'job':job,'seconds':time.perf_counter()-start,'construction':times,
               'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','calibrate','preflight','cache','train','evaluate','teacher','benchmark','export'])
    p.add_argument('--job');p.add_argument('--shard',type=int,default=0);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');args=p.parse_args()
    try:
        configure('cpu' if args.stage=='prepare' else args.device)
        job=next((j for j in jobs() if j['id']==args.job),None)
        if args.stage=='prepare':prepare()
        elif args.stage=='calibrate':calibrate()
        elif args.stage=='preflight':preflight()
        elif args.stage=='cache':cache(args.shard)
        elif args.stage=='train':train_shard(job,args.shard)
        elif args.stage=='evaluate':final_eval(job)
        elif args.stage=='teacher':teacher_eval()
        elif args.stage=='benchmark':benchmark(args.job,args.device)
        elif args.stage=='export':export(args.job)
    except Exception:
        append(ROOT/'logs/errors.jsonl',{'stage':args.stage,'job':args.job,'shard':args.shard,'error':traceback.format_exc()})
        raise
