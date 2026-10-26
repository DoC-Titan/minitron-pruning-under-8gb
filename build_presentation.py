"""Build a measured-results twelve-slide deck using installed PowerPoint."""
import json
import subprocess
import shutil
from pathlib import Path
from core import ROOT, atomic_json, sha


def main():
    s=json.loads((ROOT/'reports/SUMMARY.json').read_text());v=json.loads((ROOT/'reports/VALIDATION.json').read_text())
    assert v['status']=='PASS'
    grouped=s['aggregate'];cost=s['costs'];slides=[]
    def add(title,kicker,body,notes,figure=None):
        slides.append({'title':title,'kicker':kicker,'body':body,'notes':notes,'figure':figure})
    add('Smaller models. Stronger evidence.','STRUCTURED PRUNING + DISTILLATION | LEVEL 4',
        ['A Minitron-inspired study under an 8 GiB GPU budget','Qwen2.5-3B → physically smaller students','18 recovery runs · 3 seeds · CPU and CUDA evidence'],
        'Open with the engineering constraint, not a claim of reproducing NVIDIA scale. This is built with Qwen and inspired by Minitron. The project contribution is a controlled, auditable compression and systems study.')
    add('Does the least-damaging cut recover best?','RESEARCH QUESTION',
        ['Compare immediate damage with equal-budget recovered quality.','Test whether the answer changes with compression severity and metric.','Judge models by quality, parameters, memory and measured execution cost.'],
        'Explain that pruning loss and post-recovery quality need not rank architectures identically. Perplexity and downstream task accuracy are different objectives. Negative and metric-dependent results are scientifically useful.')
    add('A controlled experiment—not a model-size demo','PREREGISTERED DESIGN',
        ['One installed, pinned 3.086B teacher; no paid API or new model download.','Two sizes: 32 layers / FFN 9,600 and 28 layers / FFN 8,192.','Random versus informed pruning × 3 seeds; six matched CE controls.','Each run: 196,608 input tokens and 192 optimizer updates.'],
        'There are 12 KL recovery runs and six CE controls. Seeds are 42, 123 and 2026. Both methods use the same architecture within each compression level. This is not a broad hyperparameter sweep.')
    add('Physical pruning with an exact removal recipe','ALGORITHMIC CORRECTNESS',
        ['Depth: remove internal contiguous blocks; preserve boundary layers.','Informed depth: calibration NLL; informed FFN: coupled L2 magnitude.','FFN: slice gate/up rows and matching down columns—not masks.','Every layer/channel ID is saved; save/reload shape tests passed.'],
        'The Qwen gated FFN needs three coupled slices. Residual hidden width remains 2048. The depth procedure is different from the old pilot individual-layer ranking; never mix their provenance.')
    add('Sensitivity is measured before recovery','CALIBRATION ONLY',
        ['32 fixed calibration blocks; test data does not rank structures.','Different depth windows have different immediate damage.','FFN score-distribution map and exact channel selections are included.'],
        'Walk through one depth-window position and its NLL increase. The accompanying FFN map in the report is a magnitude distribution, not an individually ablated neuron causal-effect map.', 'depth_sensitivity.svg')
    add('Full-vocabulary KD without two large GPU models','RESOURCE-AWARE TRAINING',
        ['Teacher target shard → unload teacher → recover students.','Forward KL at T=2 with T² scaling; KL-only versus CE-only.','Rank-8 q/v LoRA, FP16 frozen base, FP32 adapters, checkpointing.','Atomic checkpoints each update; bounded disk and thermal guards.'],
        'Distinguish teacher targets from hidden states. Full-vocabulary logits retain the exact objective, unlike top-k coarsening. Cached shards are deleted only after all consumers have durable receipts. LoRA is not full-weight Minitron recovery.')
    lines=[]
    for r in grouped:
        lines.append(f'{r["level"]}/{r["criterion"]}/{r["objective"]}: PPL {r["test_ppl_mean"]:.1f} ± {r["test_ppl_sd"]:.1f}; PIQA {100*r["piqa_mean"]:.1f}%')
    add('Compression–quality trade-offs, not a universal winner','MEASURED THREE-SEED RESULTS',lines,
        'These are arithmetic means and sample SD over only three seeds. The graph uses mean test NLL; the body uses mean seed perplexity. The finite observed Pareto frontier is descriptive. Do not imply an unmeasured compression-ratio law.','pareto_quality.svg')
    effects=[]
    for e in s['paired_effects']:
        effects.append(f'{e["level"]}: {e["contrast"].replace("_"," ")} → ΔNLL {e["NLL_stats"]["mean"]:+.3f}, ΔPIQA {100*e["PIQA_stats"]["mean"]:+.1f}pp')
    ce_better=all(e['NLL_stats']['mean']>0 for e in s['paired_effects'] if e['contrast']=='KL_minus_CE_informed')
    effects.append('Matched CE recovery had lower test perplexity at both compression levels.' if ce_better else 'Objective effects are level-specific; lower NLL is better.')
    add('Recovery and the isolated distillation ablation','CONTROLLED SCIENTIFIC COMPARISON',effects,
        'Lower NLL is better and higher PIQA is better. A negative KL-minus-CE NLL difference favours KD; the measured positive differences here favour CE on perplexity. The small mean PIQA advantage of KL is not evidence of statistical superiority. The control matches initialization, structure, order, learning rate and tokens. This does not prove a 50-times token-efficiency claim.','recovery.svg')
    add('CPU and CUDA both execute—not just a device flag','SCIENTIFIC QUALITY + EXECUTION AUDIT',
        ['Complete tiny-transformer train → prune → distill on both devices.','Three-seed FP32 reference; same-weight device benchmark.','Real teacher and two students: matched quality inputs and PIQA scoring.','CPU FP32 versus CUDA FP16 is labelled device + precision.'],
        'The reduced model demonstrates algorithmic correctness, not 3B CPU training robustness. The real deployment comparison mixes hardware and precision, so the tiny same-weight FP32 experiment provides a clean device comparison. Byte-model and subword-model perplexities are not comparable.','tiny_matched_scaling.svg')
    ratios=[]
    for variant in ['teacher','moderate_informed_s42_kl','strong_informed_s42_kl']:
        rr=[r for r in s['deployment'] if r['variant']==variant]
        ratios.append(variant.replace('_informed_s42_kl','')+': '+', '.join(f'{r["prompt_tokens"]}→16: {r["deployment_e2e_ratio"]:.1f}×' for r in rr))
    ratios+=['2 warmups · 5 timings · medians + IQR · explicit placement','Loading, preprocessing, transfer and model execution are separate.']
    add('Where is GPU deployment practically beneficial?','MEASURED WORKLOAD SCALING',ratios,
        'These are CPU FP32/CUDA FP16 deployment ratios, not pure hardware speedups. Report observed sampled lengths only; when GPU already wins the smallest case, the exact crossover remains unknown below the sampled range. Check the matched-FP32 tiny plot for overhead-dominated cases.','cpu_gpu_scaling.svg')
    add('A recoverable run with a visible resource budget','ENGINEERING EVIDENCE',
        [f'Max main training allocation: {cost["maximum_training_allocated_gib"]:.2f} GiB.',
         f'Max sampled training GPU temperature: {cost["maximum_sampled_training_gpu_c"]}°C.',
         f'Peak target shard payload: {cost["target_peak_payload_bytes"]/2**30:.2f} GiB.',
         'Hash-verified checkpoints, corruption fallback, STOP / RESUME.','All final adapters retained; one merged representative exported.'],
        'Allocator peaks are not total board memory. Temperature is sampled, not a continuous hardware assurance; CPU temperature was unavailable. No overclocking or driver changes. A Windows fsync issue was logged and corrected before main training, with no experimental-design change.')
    add('What this project establishes—and what it does not','CONCLUSION + REPRODUCIBILITY',
        [f'Immediate NLL ranking persisted in {sum(r["initial_validation_NLL_winner"]==r["final_validation_NLL_winner"] for r in s["ranking_consistency"])}/{len(s["ranking_consistency"])} seed pairs—not universally.',
         'Executed CPU/GPU protocol, Pareto analysis, sensitivity and objective controls.','Not an official NVIDIA-scale reproduction or production chatbot.','Limits: one family/machine, 3 seeds, short context, custom PIQA, no teacher correction.','Code, raw evidence, recipes, adapters, report and reproduction guide.'],
        'Close with the actual outcomes from the tables, including any negative results. Qwen2.5-3B uses a noncommercial research license. TensorRT-LLM was conditionally excluded because the cited release requires Linux, while this environment is native Windows. Do not present PyTorch throughput as TensorRT. References: supplied Minitron technical report, question-bank p1/pp20–21, official Qwen and NVIDIA documentation.')
    assert len(slides)==12
    atomic_json(ROOT/'presentation/slides.json',slides)
    (ROOT/'presentation/SPEAKER_NOTES.md').write_text('\n\n'.join(f'## {i}. {x["title"]}\n{x["notes"]}' for i,x in enumerate(slides,1)),encoding='utf-8')
    subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'presentation/render.ps1')],check=True)
    atomic_json(ROOT/'presentation/BUILD_SOURCES.json',{'summary_sha256':sha(ROOT/'reports/SUMMARY.json'),
        'slides_sha256':sha(ROOT/'presentation/slides.json'),'slide_count':12,'claims':'Generated from validated measured JSON; no manual replacement of metrics'})
    # The workstation's pre-existing general Python has ReportLab. Keep the
    # measured scientific virtualenv unchanged; PDF rendering does no experiments.
    render_python=shutil.which('python')
    if render_python:
        available=subprocess.run([render_python,'-c','import reportlab'],capture_output=True).returncode==0
        if available:subprocess.run([render_python,str(ROOT/'render_report_pdf.py')],check=True)


if __name__=='__main__':main()
