"""All public quantitative claims are generated from measured JSON, never templates."""
import collections
import csv
import re
from html import escape
from core import *


def read(name):return json.loads((ROOT/name).read_text())


def polish_markdown(text):
    """Typography only; never transforms raw metrics, code blocks or CSV values."""
    replacements={
        'an8GiB':'an 8 GiB','gets196,608inputtokens/195,072causaltargets and192updates':'gets 196,608 input tokens / 195,072 causal targets and 192 updates',
        'rank8q/vLoRA':'rank-8 q/v LoRA','CPU has4declared':'CPU has 4 declared','a50×token-efficiency':'a 50× token-efficiency',
        'prompt lengths[':'prompt lengths [','smallest tested32tokens':'smallest tested 32 tokens','sampled lengthsnone':'sampled lengths: none',
        'A4-layer64-hidden':'A 4-layer, 64-hidden-dimension','to3layers/FFN128':'to 3 layers / FFN 128','subword3B':'subword 3B',
        'complement in0..11007':'complement in 0..11007','streamed in64-block':'streamed in 64-block',
        'plus16generated':'plus 16 generated','nominal512-position':'nominal 512-position',
        'adaptations:3B':'adaptations: 3B','rank8LoRA':'rank-8 LoRA','short128-token':'short 128-token',
        'blocks,196ktokens rather than94B':'blocks, 196k tokens rather than 94B','robustness of3B':'robustness of 3B',
        'The3B':'The 3B','Apache2.0':'Apache-2.0','all18final adapters':'all 18 final adapters','etal.(2024)':'et al. (2024)',
        'p.1andpp.20–21':'p. 1 and pp. 20–21','18matchedrecoveryruns':'18 matched recovery runs','2compressionlevels':'2 compression levels',
        '3seeds':'3 seeds','FP16deployment':'FP16 deployment','matched-FP32tinyexperiments':'matched-FP32 tiny experiments',
        'Three seeds do not establish':'Three seeds do not establish','real3Bquality':'real 3B quality','CPU/GPUscientific':'CPU/GPU scientific',
        'Three-size':'Three-size','CPU/CUDAtiny':'CPU/CUDA tiny','CPU/GPUsystem':'CPU/GPU system','12-slidepresentation':'12-slide presentation',
        'all18final':'all 18 final','Several seeds':'Several seeds','all18matched':'all 18 matched','8GiB':'8 GiB',
    }
    out=[];fenced=False
    for line in text.splitlines():
        if line.startswith('```'):fenced=not fenced;out.append(line);continue
        if not fenced:
            for old,new in replacements.items():line=line.replace(old,new)
            line=re.sub(r'\bmean(?=[+-]\d)', 'mean ', line)
            line=re.sub(r'\bSD(?=\d)', 'SD ', line)
        out.append(line)
    return '\n'.join(out)


def stats(xs):
    return {'mean':statistics.mean(xs),'sd':statistics.stdev(xs) if len(xs)>1 else 0.,'min':min(xs),'max':max(xs),'n':len(xs)}


def csvout(name,rows):
    if not rows:return
    with (ROOT/'reports'/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def scatter(name,title,xlabel,ylabel,series,logy=False):
    # Vector graphics without extra runtime installations. Each point is measured.
    colors=['#007f83','#ed8446','#5472bb','#a35691','#3a995f','#bb5656','#182d43']
    pts=[p for s in series for p in s['points']];xs=[p[0] for p in pts];ys=[math.log10(p[1]) if logy else p[1] for p in pts]
    xmin,xmax=min(xs),max(xs);ymin,ymax=min(ys),max(ys)
    dx=max((xmax-xmin)*.08,.1);dy=max((ymax-ymin)*.1,.05);xmin-=dx;xmax+=dx;ymin-=dy;ymax+=dy
    X=lambda x:85+(x-xmin)/(xmax-xmin)*650
    Y=lambda y:390-((math.log10(y) if logy else y)-ymin)/(ymax-ymin)*300
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="490" viewBox="0 0 1000 490">',
         '<rect width="1000" height="490" fill="white"/>',f'<text x="40" y="32" font-family="Arial" font-size="21" fill="#172d43">{escape(title)}</text>']
    for i in range(6):
        x=xmin+(xmax-xmin)*i/5;y=ymin+(ymax-ymin)*i/5
        out+=[f'<path d="M{X(x):.2f},85 V390" stroke="#e4ebee"/><text x="{X(x):.2f}" y="412" text-anchor="middle" font-family="Arial" font-size="12">{x:.2f}</text>',
              f'<path d="M85,{390-i*60} H735" stroke="#e4ebee"/><text x="74" y="{394-i*60}" text-anchor="end" font-family="Arial" font-size="12">{(10**y if logy else y):.2f}</text>']
    for i,s in enumerate(series):
        color=s.get('color',colors[i%len(colors)])
        if s.get('line') and len(s['points'])>1:
            coords=' '.join(f'{X(x):.2f},{Y(y):.2f}' for x,y in s['points'])
            dash=' stroke-dasharray="6 4"' if s.get('dashed') else ''
            out.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
        for x,y in s['points']:out.append(f'<circle cx="{X(x):.2f}" cy="{Y(y):.2f}" r="4.5" fill="{color}"/>')
        out.append(f'<rect x="760" y="{95+i*29}" width="10" height="10" fill="{color}"/><text x="777" y="{105+i*29}" font-family="Arial" font-size="12">{escape(s["label"])}</text>')
    out+=[f'<text x="400" y="452" text-anchor="middle" font-family="Arial" font-size="15">{escape(xlabel)}</text>',
          f'<text x="23" y="240" transform="rotate(-90 23 240)" text-anchor="middle" font-family="Arial" font-size="15">{escape(ylabel)}</text>','</svg>']
    (ROOT/'figures'/name).write_text('\n'.join(out),encoding='utf-8')


def sensitivity_figures():
    sens=read('metrics/sensitivity.json');series=[]
    for level,rows in sens['rankings'].items():series.append({'label':level,'points':[(r['start'],r['delta_nll']) for r in rows],'line':True})
    scatter('depth_sensitivity.svg','Contiguous depth-window sensitivity (calibration only)','First removed original layer (zero-based)','Increase in calibration NLL',series)
    scores=checked_load(ROOT/'recipes/ffn_magnitude_scores.pt');out=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="490"><rect width="1000" height="490" fill="white"/>',
        '<text x="35" y="30" font-family="Arial" font-size="21">FFN magnitude sensitivity distribution by layer</text>']
    q=torch.quantile(scores,torch.linspace(0,1,11),dim=1).T
    low=float(q.min());high=float(q.max())
    for i,row in enumerate(q):
        for j,v in enumerate(row):
            intensity=(float(v)-low)/(high-low);r=int(235-215*intensity);g=int(245-115*intensity);b=int(247-112*intensity)
            out.append(f'<rect x="{100+i*22}" y="{355-j*25}" width="21" height="24" fill="rgb({r},{g},{b})"/>')
        if i%4==0:out.append(f'<text x="{104+i*22}" y="400" font-size="12" font-family="Arial">{i}</text>')
    out+=['<text x="100" y="432" font-family="Arial" font-size="15">Original layer →; vertical axis: channel score quantiles 0% (bottom) to100% (top)</text>',
          '<text x="100" y="462" font-family="Arial" font-size="13">Coupled gate/up row and down column L2 energy; descriptive scores, not causal per-channel ablation.</text>','</svg>']
    (ROOT/'figures/ffn_sensitivity_map.svg').write_text('\n'.join(out),encoding='utf-8')


def main():
    assert read('reports/VALIDATION.json')['status']=='PASS'
    results=[read(f'metrics/{j["id"]}_final.json') for j in jobs()];teacher=read('metrics/teacher_final.json');groups=collections.defaultdict(list);rows=[]
    for r in results:
        j=r['job'];groups[(j['level'],j['criterion'],j['objective'])].append(r)
        rows.append(dict(id=j['id'],level=j['level'],criterion=j['criterion'],objective=j['objective'],seed=j['seed'],
                         parameters=r['parameters'],reduction_pct=r['parameter_reduction_percent'],test_nll=r['test']['nll'],test_ppl=r['test']['perplexity'],
                         piqa_accuracy=r['piqa']['accuracy'],initial_val_nll=r['recovery'][0]['validation']['nll'],final_val_nll=r['validation']['nll'],
                         input_tokens=r['totals']['input_tokens'],updates=r['totals']['updates'],training_seconds=r['totals']['model_execution_seconds'],
                         training_peak_gib=r['totals']['training_peak_allocated_gib'],
                         gpu_generated_tokens_per_second=r['generation_benchmark']['generated_tokens_per_second'],
                         gpu_generation_peak_allocated_gib=r['generation_benchmark']['memory']['allocated_gib']))
    csvout('PER_SEED_RESULTS.csv',rows)
    aggregate=[]
    for key,rs in groups.items():
        agg={'level':key[0],'criterion':key[1],'objective':key[2],'parameters':rs[0]['parameters'],'reduction_pct':rs[0]['parameter_reduction_percent']}
        for field,values in [('test_nll',[r['test']['nll'] for r in rs]),('test_ppl',[r['test']['perplexity'] for r in rs]),
                             ('piqa',[r['piqa']['accuracy'] for r in rs]),('initial_val_nll',[r['recovery'][0]['validation']['nll'] for r in rs]),
                             ('final_val_nll',[r['validation']['nll'] for r in rs])]:
            for name,value in stats(values).items():agg[field+'_'+name]=value
        aggregate.append(agg)
    csvout('AGGREGATE_RESULTS.csv',aggregate)
    effects=[];lookup={r['job']['id']:r for r in results}
    for level in C['structures']:
        for contrast in ['informed_minus_random_KL','KL_minus_CE_informed']:
            diffs=[];task=[];initial=[]
            for seed in C['seeds']:
                a=lookup[f'{level}_informed_s{seed}_kl'];b=lookup[f'{level}_random_s{seed}_kl'] if contrast.startswith('informed') else lookup[f'{level}_informed_s{seed}_ce']
                diffs.append(a['test']['nll']-b['test']['nll']);task.append(a['piqa']['accuracy']-b['piqa']['accuracy'])
                initial.append(a['recovery'][0]['validation']['nll']-b['recovery'][0]['validation']['nll'])
            effects.append({'level':level,'contrast':contrast,'paired_test_NLL_differences':diffs,'NLL_stats':stats(diffs),
                            'paired_PIQA_accuracy_differences':task,'PIQA_stats':stats(task),'initial_validation_NLL_differences':initial})
    atomic_json(ROOT/'reports/PAIRED_EFFECTS.json',effects)
    ranking=[]
    def winner(a,b,lower=True):
        if abs(a-b)<1e-8:return 'tie'
        return 'informed' if ((a<b) if lower else (a>b)) else 'random'
    for level in C['structures']:
        for seed in C['seeds']:
            a=lookup[f'{level}_informed_s{seed}_kl'];b=lookup[f'{level}_random_s{seed}_kl']
            ranking.append({'level':level,'seed':seed,
                'initial_validation_NLL_winner':winner(a['recovery'][0]['validation']['nll'],b['recovery'][0]['validation']['nll']),
                'final_validation_NLL_winner':winner(a['validation']['nll'],b['validation']['nll']),
                'final_test_NLL_winner':winner(a['test']['nll'],b['test']['nll']),
                'final_PIQA_winner':winner(a['piqa']['accuracy'],b['piqa']['accuracy'],False)})
    atomic_json(ROOT/'reports/RANKING_CONSISTENCY.json',ranking)
    # Explicit observed finite-point frontier: teacher and equal-budget group means.
    points=[{'label':'teacher','parameters':teacher['parameters'],'nll':teacher['test']['nll'],'piqa':teacher['piqa']['accuracy']}]
    points += [{'label':f'{r["level"]}/{r["criterion"]}/{r["objective"]}','parameters':r['parameters'],'nll':r['test_nll_mean'],'piqa':r['piqa_mean']} for r in aggregate]
    frontier=[p for p in points if not any(q['parameters']<=p['parameters'] and q['nll']<=p['nll'] and (q['parameters']<p['parameters'] or q['nll']<p['nll']) for q in points)]
    task_frontier=[p for p in points if not any(q['parameters']<=p['parameters'] and q['piqa']>=p['piqa'] and (q['parameters']<p['parameters'] or q['piqa']>p['piqa']) for q in points)]
    atomic_json(ROOT/'reports/PARETO_POINTS.json',{'points':points,'nll_frontier':frontier,'piqa_frontier':task_frontier,'caveat':'Observed finite points, same fixed recovery budget; teacher is untrained control. Dashed connections identify observed frontier membership, not an interpolated ratio-quality law.'})
    scatter('pareto_quality.svg','Compression–quality: measured NLL frontier','Parameters (billions)','Test NLL (lower is better)',
            [{'label':p['label'],'points':[(p['parameters']/1e9,p['nll'])]} for p in points]+
            [{'label':'Observed frontier','points':[(p['parameters']/1e9,p['nll']) for p in sorted(frontier,key=lambda p:p['parameters'])],'line':True,'dashed':True,'color':'#213647'}])
    scatter('pareto_task.svg','Compression–quality: measured PIQA frontier','Parameters (billions)','PIQA accuracy (%) — higher is better',
            [{'label':p['label'],'points':[(p['parameters']/1e9,p['piqa']*100)]} for p in points]+
            [{'label':'Observed frontier','points':[(p['parameters']/1e9,p['piqa']*100) for p in sorted(task_frontier,key=lambda p:p['parameters'])],'line':True,'dashed':True,'color':'#213647'}])
    recovery_series=[]
    for key,rs in groups.items():
        pts=[(r['input_tokens']/1000,statistics.mean(x['recovery'][i]['validation']['nll'] for x in rs)) for i,r in enumerate(rs[0]['recovery'])]
        recovery_series.append({'label':'/'.join(key),'points':pts,'line':True})
    scatter('recovery.svg','Equal-budget recovery (three-seed means)','Input training tokens (thousands)','Validation NLL',recovery_series)
    sensitivity_figures()
    benchrows=[];qualityrows=[]
    for variant in ['teacher','moderate_informed_s42_kl','strong_informed_s42_kl']:
        cpu=read(f'metrics/benchmark_{variant}_cpu.json');gpu=read(f'metrics/benchmark_{variant}_cuda.json')
        qualityrows.append({'variant':variant,'cpu_fp32_ppl':cpu['quality']['test']['perplexity'],'gpu_fp16_ppl':gpu['quality']['test']['perplexity'],
                'nll_difference_gpu_minus_cpu':gpu['quality']['test']['nll']-cpu['quality']['test']['nll'],
                'cpu_piqa':cpu['quality']['piqa']['accuracy'],'gpu_piqa':gpu['quality']['piqa']['accuracy'],
                'paired_prediction_disagreements':sum(a['prediction']!=b['prediction'] for a,b in zip(cpu['quality']['piqa']['details'],gpu['quality']['piqa']['details']))})
        for a,b in zip(cpu['cases'],gpu['cases']):
            benchrows.append({'variant':variant,'prompt_tokens':a['prompt_tokens'],'output_tokens':16,
                'cpu_model_seconds':a['median']['execution_seconds'],'cuda_model_seconds':b['median']['execution_seconds'],
                'cpu_request_e2e_seconds':a['median']['request_e2e_seconds'],'cuda_request_e2e_seconds':b['median']['request_e2e_seconds'],
                'deployment_model_ratio':a['median']['execution_seconds']/b['median']['execution_seconds'],
                'deployment_e2e_ratio':a['median']['request_e2e_seconds']/b['median']['request_e2e_seconds'],
                'cpu_load_reconstruct_seconds':cpu['model_load_seconds'],'cuda_load_reconstruct_seconds':gpu['model_load_seconds'],
                'cpu_execution_iqr_seconds':a['execution_iqr_seconds'],'cuda_execution_iqr_seconds':b['execution_iqr_seconds']})
    csvout('CPU_GPU_DEPLOYMENT.csv',benchrows);csvout('CPU_GPU_QUALITY.csv',qualityrows)
    scatter('cpu_gpu_scaling.svg','Real-model deployment: CPU FP32 versus CUDA FP16','Prompt tokens (16 generated)','Median prefill + decode seconds',
            [{'label':variant+'/'+device,'points':[(r['prompt_tokens'],r[device+'_model_seconds']) for r in benchrows if r['variant']==variant],'line':True}
             for variant in ['teacher','moderate_informed_s42_kl','strong_informed_s42_kl'] for device in ['cpu','cuda']],logy=True)
    tc=read('metrics/tiny_cpu.json');tg=read('metrics/tiny_cuda.json');tinybench=[]
    for a,b in zip(tc['benchmark']['cases'],tg['benchmark']['cases']):
        tinybench.append({'prompt_tokens':a['prompt_tokens'],'cpu_seconds':a['median_seconds'],'cuda_seconds':b['median_seconds'],
                          'model_ratio':a['median_seconds']/b['median_seconds'],'e2e_ratio':a['median_e2e_seconds']/b['median_e2e_seconds']})
    csvout('TINY_MATCHED_FP32_SCALING.csv',tinybench)
    scatter('tiny_matched_scaling.svg','Matched FP32 tiny transformer: same weights and inputs','Prompt tokens (16 generated)','Median prefill + decode seconds',
           [{'label':device,'points':[(r['prompt_tokens'],r[device+'_seconds']) for r in tinybench],'line':True} for device in ['cpu','cuda']])
    tinyrows=[]
    for dev,rr in [('cpu',tc),('cuda',tg)]:
        for r in rr['rows']:
            tinyrows.append({'device':dev,'seed':r['seed'],'criterion':r['criterion'],'objective':r['objective'],'parameters':r['parameters'],
                            'initial_val_ppl':r['result']['initial_validation']['perplexity'],'final_val_ppl':r['result']['final_validation']['perplexity'],
                            'test_ppl':r['result']['test']['perplexity'],'training_seconds':r['result']['training_seconds'],'input_tokens':r['result']['input_tokens']})
    csvout('TINY_CPU_GPU_RESULTS.csv',tinyrows)
    caches=[json.loads(p.read_text()) for p in (ROOT/'cache').glob('full_logits_*.json')]
    logs=[]
    for p in (ROOT/'logs').glob('*_s*.jsonl'):
        for line in p.read_text().splitlines():
            try:logs.append(json.loads(line))
            except ValueError:pass
    temps=[x['gpu_sample']['temperature_c'] for x in logs if 'temperature_c' in x.get('gpu_sample',{})]
    stages=[json.loads(line) for line in (ROOT/'logs/stage_costs.jsonl').read_text().splitlines()]
    exported=read('model/EXPORT_COMPLETE.json')
    costs={'data_preparation_seconds':read('data/manifest.json')['prepare_seconds'],
           'target_inference_transfer_write_seconds':sum(r['inference_transfer_write_seconds'] for r in caches),
           'target_model_load_seconds':sum(r['load_seconds'] for r in caches),'target_peak_payload_bytes':max(r['actual_bytes'] for r in caches),
           'total_teacher_targets_generated_bytes':sum(r['actual_bytes'] for r in caches),'calibration_seconds':read('metrics/sensitivity.json')['seconds'],
           'calibration_model_load_seconds':read('metrics/sensitivity.json')['load_seconds'],
           'main_model_execution_seconds':sum(r['totals']['model_execution_seconds'] for r in results),
           'main_target_transfer_seconds':sum(r['totals']['target_read_transfer_seconds'] for r in results),
           'main_reconstruction_seconds':sum(r['totals']['construction_seconds'] for r in results),
           'main_validation_seconds':sum(r['totals']['validation_seconds'] for r in results),
           'final_quality_and_generation_seconds':sum(r['evaluation_seconds'] for r in results),
           'final_reconstruction_and_merge_seconds':sum(sum(r['construction'].values()) for r in results),
           'model_export_write_seconds':exported['seconds'],
           'model_export_reconstruction_seconds':sum(exported['construction'].values()),
           'tiny_model_execution_seconds':sum(r['result']['training_seconds'] for rr in [tc,tg] for r in rr['rows']),
           'checkpoint_seconds_approx':sum(r['totals']['checkpoint_seconds'] for r in results),
           'stage_wall_seconds':sum(r['seconds'] for r in stages),'maximum_sampled_training_gpu_c':max(temps) if temps else None,
           'maximum_training_allocated_gib':max(r['totals']['training_peak_allocated_gib'] for r in results),
           'notes':['Stage wall totals include process/import/monitor/IO overhead; not equal to sum of model-only timings.',
                    'Checkpoint timing cumulative counter excludes the most recent write due to checkpoint serialization ordering; approximate, not exact.',
                    'Power-cut/replayed steps may add wall cost; logical tokens count only committed updates.',
                    'load_inclusive_first_request_seconds in raw benchmark is load plus first timed WARM request, a proxy, not measured cold-first-request latency.']}
    atomic_json(ROOT/'reports/PHASE_COSTS.json',costs)
    # Publishable software/hardware record with no username or absolute command lines.
    env=read('environment.json');env.pop('nvidia_smi',None)
    atomic_json(ROOT/'reports/HARDWARE_SOFTWARE.json',env)
    summary={'teacher':{'parameters':teacher['parameters'],'ppl':teacher['test']['perplexity'],'piqa':teacher['piqa']['accuracy']},
             'aggregate':aggregate,'paired_effects':effects,'deployment':benchrows,'quality_comparison':qualityrows,
             'tiny_benchmark':tinybench,'costs':costs,'main_runs':len(results),'seeds':C['seeds'],'ranking_consistency':ranking}
    atomic_json(ROOT/'reports/SUMMARY.json',summary)
    report(summary,rows,frontier)


def report(s,rows,frontier):
    objective_effects=[e for e in s['paired_effects'] if e['contrast']=='KL_minus_CE_informed']
    ce_better_both=all(e['NLL_stats']['mean']>0 for e in objective_effects)
    task_effects=[e['PIQA_stats']['mean']*100 for e in objective_effects]
    t=s['teacher'];lines=['# Structured Pruning and Distillation under an8GiB GPU Budget','',
        '## Executive result',f'Executed **{s["main_runs"]} full-size recovery conditions**, two physical compression levels, three seeds, matched CE controls, CPU/CUDA references and measured workload scaling.',
        f'Teacher: {t["parameters"]:,} parameters; test PPL **{t["ppl"]:.2f}**; PIQA200 **{100*t["piqa"]:.1f}%**. These are local measurements, not NVIDIA scores.','',
        '## Scientific design','See [locked plan](../EXPERIMENT_PLAN.md), config.json and exact recipes. Every main run gets196,608inputtokens/195,072causaltargets and192updates; full-vocabulary T=2 KL or ordinary CE, same rank8q/vLoRA budget.',
        'The informed depth criterion is contiguous-window calibration NLL, not the earlier pilot\'s individual-layer ranking. Both criteria preserve boundary layers. Random architecture and training vary with seed; informed architecture is shared and training initialization/order vary.',
        'No automatic hyperparameter sweep or test-based checkpoint selection was performed. The exported strong/informed/seed42 model is a predeclared representative, not a claimed best model.','',
        '## Three-seed final comparison','Values are arithmetic mean ± sample SD across three seeds; no significance or population-generalization claim. PPL is mean of seed PPLs, not exp(mean NLL).','',
        '| Structure / criterion / objective | Parameters B | Reduction | Test PPL | PIQA accuracy |',
        '|---|---:|---:|---:|---:|']
    for r in s['aggregate']:
        lines.append(f'| {r["level"]}/{r["criterion"]}/{r["objective"]} | {r["parameters"]/1e9:.3f} | {r["reduction_pct"]:.2f}% | {r["test_ppl_mean"]:.2f} ± {r["test_ppl_sd"]:.2f} | {100*r["piqa_mean"]:.1f}% ± {100*r["piqa_sd"]:.1f}pp |')
    lines+=['','## Does minimum immediate damage predict best recovery?',
            'Compare validation NLL before/after at the same structure and seed; downstream task ordering is a separate question. Lower NLL is better, higher PIQA is better. Do not extrapolate one criterion universally.','']
    for e in s['paired_effects']:
        initial=e['initial_validation_NLL_differences'];v=e['NLL_stats'];q=e['PIQA_stats']
        lines.append(f'- **{e["level"]}, {e["contrast"]}:** paired test ΔNLL mean{v["mean"]:+.4f}, SD{v["sd"]:.4f}, range[{v["min"]:+.4f},{v["max"]:+.4f}]; paired ΔPIQA{q["mean"]*100:+.2f}pp. Initial validation ΔNLLs: '+', '.join(f'{x:+.4f}' for x in initial)+'.')
    for level in C['structures']:
        rr=[r for r in s['ranking_consistency'] if r['level']==level]
        same_val=sum(r['initial_validation_NLL_winner']==r['final_validation_NLL_winner'] for r in rr)
        same_test=sum(r['initial_validation_NLL_winner']==r['final_test_NLL_winner'] for r in rr)
        same_task=sum(r['initial_validation_NLL_winner']==r['final_PIQA_winner'] for r in rr)
        lines.append(f'\n**Direct answer at {level} compression:** the criterion with lower immediate validation NLL also won final validation NLL in **{same_val}/3** seed pairs, final test NLL in **{same_test}/3**, and final PIQA in **{same_task}/3** (ties are retained explicitly). This describes these measured pairs, not a universal ranking law.')
    lines+=['','### Controlled objective result',
        ('At both compression levels, the matched CE-only recovery achieved lower WikiText test perplexity than KL-only recovery at the same token budget.' if ce_better_both else 'The objective comparison is reported separately for each compression level; do not assume one objective wins universally.')+
        f' The measured KL-minus-CE mean PIQA effects span {min(task_effects):+.2f} to {max(task_effects):+.2f} percentage points; this small shared task set does not establish statistical superiority.',
        'A distillation advantage must be established by measurements, not presumed. These experiments do not isolate why the objectives differ: finite token budget, adapter-only capacity, temperature and an uncorrected teacher are plausible hypotheses, not established causes. No post-hoc tuning or removal of unfavorable findings is performed.']
    lines+=['','The isolated KL-minus-CE contrast changes only the objective. It does not establish NVIDIA-scale teacher-correction, full-weight training or a50×token-efficiency gain.',
            '![Recovery](../figures/recovery.svg)','![Pareto](../figures/pareto_quality.svg)','![Task Pareto](../figures/pareto_task.svg)','',
            '## CPU / GPU execution and scientific quality',
            'Real-model CPU FP32 versus CUDA FP16 is a **deployment device+precision comparison**, not isolated hardware acceleration. Same model recipes, rounded merged weights, held-out subset and PIQA scoring are used. CPU has4declared threads, leaving resources for the desktop. No fallback is allowed.','',
            'This paired quality table uses the same first **16 test blocks (2,032 causal targets) and 32 PIQA examples** on both devices. The main model table instead uses 256 blocks and 200 examples. Its PPL is therefore not numerically interchangeable with this smaller, harder subset.','',
            '| Model | CPU FP32 PPL | CUDA FP16 PPL | ΔNLL GPU−CPU | CPU / GPU PIQA | Disagreements /32 |','|---|---:|---:|---:|---:|---:|']
    for r in s['quality_comparison']:lines.append(f'| {r["variant"]} | {r["cpu_fp32_ppl"]:.3f} | {r["gpu_fp16_ppl"]:.3f} | {r["nll_difference_gpu_minus_cpu"]:+.5f} | {100*r["cpu_piqa"]:.1f}% / {100*r["gpu_piqa"]:.1f}% | {r["paired_prediction_disagreements"]} |')
    lines+=['','Two warmups and five synchronized timed requests per size; raw timings and IQR are retained. The request timer separates tokenization, host/device transfer, prefill+decode and output decode. Reconstruction/loading is separate. Loading is not an OS-cold-cache measurement.','',
            '| Model | Prompt→output | CPU model s | CUDA model s | Deployment model ratio | Request E2E ratio |','|---|---:|---:|---:|---:|---:|']
    for r in s['deployment']:lines.append(f'| {r["variant"]} | {r["prompt_tokens"]}→16 | {r["cpu_model_seconds"]:.3f} | {r["cuda_model_seconds"]:.3f} | {r["deployment_model_ratio"]:.2f}× | {r["deployment_e2e_ratio"]:.2f}× |')
    lines+=['','### Observed benefit scale']
    for variant in sorted(set(r['variant'] for r in s['deployment'])):
        rs=[r for r in s['deployment'] if r['variant']==variant];win=[r['prompt_tokens'] for r in rs if r['deployment_e2e_ratio']>1]
        lines.append(f'- {variant}: CUDA deployment request benefit observed at prompt lengths{win or "none"}. '+('Already beneficial at smallest tested32tokens; exact crossover is below/unknown relative to sampled range.' if len(win)==3 else 'No exact crossover inferred between sampled points.'))
    wins=[r['prompt_tokens'] for r in s['tiny_benchmark'] if r['e2e_ratio']>1]
    lines+=['',f'**Clean hardware comparison:** the tiny reference uses identical learned weights and FP32 on both devices. GPU request benefit observed at sampled lengths{wins or "none"}; the absence of benefit is a valid result, not a failed benchmark.',
            '![Deployment scaling](../figures/cpu_gpu_scaling.svg)','![Matched FP32 scaling](../figures/tiny_matched_scaling.svg)','',
            '## Reduced no-GPU reference',
            'A4-layer64-hidden byte-level Qwen transformer is trained on local WikiText train text; physically pruned to3layers/FFN128, recovered by KL/CE on CPU and CUDA across the same three seeds. Full-weight tiny recovery is used, unlike main LoRA recovery. Complete per-seed quality, recipes, counts and timings: TINY_CPU_GPU_RESULTS.csv. Byte-level PPL must never be numerically compared to subword3B PPL.','',
            '## Sensitivity, structural removal and Pareto frontier',
            '![Depth sensitivity](../figures/depth_sensitivity.svg)','![FFN score map](../figures/ffn_sensitivity_map.svg)',
            'Every recipe stores exact retained original layer IDs and coupled FFN IDs. Removed channels are the complement in0..11007, not zeros/masks. FFN map shows magnitude distributions; only depth map is causal ablation sensitivity. The finite observed NLL Pareto frontier is: '+', '.join(p['label'] for p in frontier)+'. Lines do not imply unmeasured architectures.','',
            '## Workflow cost and resource safety',
            'Full-vocabulary targets were streamed in64-block shards, consumed by all conditions, then their payloads removed only after durable checkpoint receipts. Payload hashes/manifests remain. No hidden-state targets or approximate top-k objective were substituted.','',
            '```json',json.dumps(s['costs'],indent=2),'```','',
            'See PHASE_COSTS.json and logs/stage_costs.jsonl. Stage-wall time includes Python imports, monitoring, checkpointing and other overhead, and is not reported as estimator-only time. Main execution includes forward/backward/optimizer; target transfer is timed separately. Validation costs are not folded into optimizer time. Checkpoint-time counter is approximate as disclosed.','',
            '## NVIDIA paper comparison and boundaries',
            '- Shared principles: structural trimming, calibration ranking, random baseline, teacher-logit forward KL, token-budget recovery analysis.',
            '- Deliberate adaptations:3B teacher, joint depth+FFN (paper often separates depth/width), rank8LoRA rather than full-weight main recovery, short128-token blocks,196ktokens rather than94B–380B, and no teacher correction/NAS/alignment.',
            '- Paper reports Minitron width/depth downstream differences and teacher correction. Our measurements cannot establish those large-scale conclusions; our CE ablation only tests the local objective contrast.',
            '- PyTorch CUDA is executed. TensorRT-LLM is conditionally excluded with official native-Windows compatibility evidence in BACKEND_SUPPORT_AUDIT.md. No TensorRT throughput claim. RAPIDS/cuML fallback: N/A.','',
            '## Limitations and threats to validity',
            'Only three seeds, two levels, one model family, one machine and a short recovery budget. Fixed calibration, custom PIQA200 and previously disclosed WikiText test partition are not broad external validation. No semantic/near-duplicate or pretraining-contamination guarantee. Exact-block/source split checks are narrower. PIQA examples and text blocks are shared across seeds; do not treat all predictions as independent replicates. Desktop load, CPU thread budget, uncontrolled OS cache and thermal state affect timings. GPU system samples are intermittent; allocator peaks are not total board memory. CPU temperature is not available from a verified sensor. The CPU tiny reference validates mechanics/trends, not robustness of3B training on CPU. Its 512-token prompt plus16generated tokens exceeds the nominal512-position configuration; execution succeeded with rotary attention, but this is not a validated long-context quality result (see PROTOCOL_AMENDMENTS.md). No API teacher or quantization claim; parameter reduction is separated from arithmetic precision. The3B model license is noncommercial, not Apache2.0.','',
            '## Reproducibility, portfolio and evidence ledger',
            '[Reproduction commands](../REPRODUCE.md), [model card](../MODEL_CARD.md), [data card](../DATA_CARD.md), [validation receipt](VALIDATION.json), [requirements ledger](REQUIREMENTS_LEDGER.md), pinned configs/recipes, all18final adapters and raw per-seed/per-example JSON. Saved optimizer checkpoints and STOP/RESUME commands tolerate normal interruption but cannot protect against all disk/power faults. The sealed earlier submission is preserved as the pilot, not overwritten.','',
            '## References',
            'Sreenivas etal.(2024), LLM Pruning and Distillation in Practice: The Minitron Approach, supplied technical report; https://arxiv.org/abs/2408.11796 .',
            'Muralidharan etal.(2024), Compact Language Models via Pruning and Knowledge Distillation; https://arxiv.org/abs/2407.14679 .',
            'Qwen2.5-3B model card/license: https://huggingface.co/Qwen/Qwen2.5-3B .',
            'WikiText: https://huggingface.co/datasets/Salesforce/wikitext . PIQA: https://github.com/ybisk/ybisk.github.io/tree/master/piqa .',
            'Supplied Advanced Python HOTS & Research Project Question Bank, p.1andpp.20–21; linked resources remain their owners\' copyrighted materials. Do not redistribute supplied PDFs as project-authored content.']
    (ROOT/'reports/TECHNICAL_REPORT.md').write_text(polish_markdown('\n'.join(lines)),encoding='utf-8')
    ledger=['# Requirement-to-evidence ledger','',
            '| Requirement | Executed evidence / artifact | Scope |','|---|---|---|',
            '| Open1B–8Bmodel/licensing/VRAM | HARDWARE_SOFTWARE.json, MODEL_CARD.md, preflight receipt |3Bnoncommercial research |',
            '| Two physical dimensions, exact groups | recipes/*.json, unit shape/save/reload test | Depth + coupled FFN |',
            '| Random versus sensitivity/magnitude | PER_SEED_RESULTS.csv, sensitivity.json | Matched architectures within each level |',
            '| Teacher distillation/recovery versus tokens | Full-vocab cache manifests, per-run recovery logs, recovery.svg |196608tokens each; LoRA |',
            '| PPL/task/parameters/memory/throughput | Raw final metrics, CPU_GPU*.csv, phase costs | Custom PIQA and short-context PPL |',
            '| Immediate damage versus final ranking | PAIRED_EFFECTS.json and initial/final NLL columns | Metric/level/seed-specific |',
            '| Executed reduced CPU prune+distill | tiny_cpu.json, TINY_CPU_GPU_RESULTS.csv | Tiny byte LM, not3BCPUtraining |',
            '| Actual NVIDIA execution/fallback audit | Parameter-device assertions, CUDA timings, HARDWARE_SOFTWARE.json | No auto fallback; cuML N/A |',
            '| Multi-size warm repeated median timings | benchmark*.json, tiny benchmark JSON |2warmups/5timings,3real sizes/5tiny sizes |',
            '| CPU/GPUscientific comparison | CPU_GPU_QUALITY.csv; matchedFP32tinyquality | Deployment precision confound explicitly labelled |',
            '| Several seeds | Three seeds in18mainruns and CPU/CUDA tiny runs | Seed42timing representative, not replicated hardware sessions |',
            '| Phase-separated costs, complete metadata | PHASE_COSTS.json, raw benchmarks, HARDWARE_SOFTWARE.json | Checkpoint counter approximate; no CPU temperature |',
            '| Observed GPU practical benefit scale | CPU_GPU_DEPLOYMENT.csv, TINY_MATCHED_FP32_SCALING.csv | Sampled benefit only; no exact unsampled crossover |',
            '| Pareto / sensitivity / distillation ablation | figures/*.svg, PARETO_POINTS.json, PAIRED_EFFECTS.json | KL-only versus CE-only matched control |',
            '| TensorRT-LLM if supported | BACKEND_SUPPORT_AUDIT.md | Native-Windows cited release unsupported; not executed |',
            '| Durable execution and reproducibility | Atomic hashed checkpoints, corruption-fallback unit test, RESUME.ps1 | No UPS or hardware-fault guarantee |',
            '| New12-slidepresentation | presentation/ after build validation | Generated strictly from measured summary |',
            '| GitHub publication | publish_receipt.local.json only after verified remote | Pending authentication/publication until receipt exists |']
    (ROOT/'reports/REQUIREMENTS_LEDGER.md').write_text(polish_markdown('\n'.join(ledger)),encoding='utf-8')
    head=['# Pruning under an8GiB GPU budget','',
          '**A reproducible, Minitron-inspired structured-compression and systems study—built with Qwen.**','',
          f'18matchedrecoveryruns · 2compressionlevels · 3seeds · CPU/CUDAexecution · full-vocabularyKD · checkpointed execution','',
          '![Compression quality](figures/pareto_quality.svg)','',
          '## Start here','- [Technical report and measured conclusions](reports/TECHNICAL_REPORT.md)',
          '- [All requirement evidence](reports/REQUIREMENTS_LEDGER.md)',
          '- [Per-seed results](reports/PER_SEED_RESULTS.csv) and [paired objective/criterion effects](reports/PAIRED_EFFECTS.json)',
          '- [CPU/GPU execution and precision disclosure](reports/CPU_GPU_DEPLOYMENT.csv)',
          '- [12-slide presentation, PDF and speaker notes](presentation/)',
          '- [Reproduce / resume / stop](REPRODUCE.md), [model card](MODEL_CARD.md), [data card](DATA_CARD.md)',
          '- [Preregistered design](EXPERIMENT_PLAN.md) and [validation](reports/VALIDATION.json)','',
          '## Engineering highlights','Physical depth/FFN trimming, seed-matched CE controls, full-vocabulary logit streaming with bounded disk, low-rank recovery, atomic hash-verified checkpoints, corruption recovery tests, explicit device audits, resource guardrails, robust timing and evidence-derived reporting.','',
          '## Honest boundaries','Not an official NVIDIA reproduction, a production chatbot, a TensorRT benchmark or a claim that larger models are unnecessary. Main recovery is adapter-only and short-budget. CPU FP32/CUDA FP16deployment ratios include precision effects; matched-FP32tinyexperiments isolate device. Three seeds do not establish universal rankings. Qwen2.5-3B has a noncommercial research license.','',
          'Large weights, optimizer states, cached targets, virtual environments and supplied copyrighted PDFs are intentionally excluded from GitHub. Exact recipes, all final adapters, raw metrics and regeneration instructions are included.']
    takeaways=['## What the measurements actually show','',
        '- Physical parameter reductions of **'+' and '.join(f'{v:.2f}%' for v in sorted(set(r['reduction_pct'] for r in s['aggregate'])))+'**, not quantization or sparse masks.',
        ('- **Distillation did not automatically win:** matched CE-only recovery achieved lower test perplexity at both sizes. The small KL advantage on PIQA is not a statistical-superiority claim.' if ce_better_both else '- **Matched objective ablation:** see the per-level CE-versus-KL effects; do not assume either objective wins universally.'),
        f'- The lower-immediate-NLL criterion retained the final validation-NLL lead in **{sum(r["initial_validation_NLL_winner"]==r["final_validation_NLL_winner"] for r in s["ranking_consistency"])}/{len(s["ranking_consistency"])}** matched seed pairs—not universally.',
        '- Real-model CPU FP32 / CUDA FP16 deployment ratios include precision effects; see the measured per-size table. On the same-weight **FP32 tiny reference, '+('CPU was faster at every sampled size' if all(r['model_ratio']<1 for r in s['tiny_benchmark']) else 'device benefit depends on the sampled workload')+'**.',
        f'- Main training peak allocation: **{s["costs"]["maximum_training_allocated_gib"]:.2f} GiB**; maximum sampled GPU temperature: **{s["costs"]["maximum_sampled_training_gpu_c"]}°C**; streamed target payload about **{s["costs"]["target_peak_payload_bytes"]/2**30:.2f} GiB**.',
        '', 'The contribution is controlled evidence and reliable engineering—not a claim of reproducing NVIDIA H100-scale quality.','']
    pos=head.index('## Start here');head[pos:pos]=takeaways
    (ROOT/'README.md').write_text(polish_markdown('\n'.join(head)),encoding='utf-8')


if __name__=='__main__':main()
