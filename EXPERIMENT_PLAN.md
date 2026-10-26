# Pruning under an 8 GiB budget: preregistered local extension

Status: implementation / feasibility tests, **not completed results**.

## Scope and claims
The question bank's mandatory protocol (p.1) and Level 4 project 2 (pp.20–21)
are the specification. The supplied NVIDIA Minitron technical report is a research
anchor, not a requirement to reproduce its 94B–380B-token/H100 training scale.
This study is built with Qwen2.5-3B BASE under the noncommercial Qwen Research
License. It is not an official NVIDIA implementation or an instruct assistant.

Research question: does minimum calibration damage predict best equal-budget
post-distillation quality, and does the answer depend on compression or metric?

## Locked main matrix
- Teacher: existing, pinned 3,085,938,688-parameter Qwen2.5-3B.
- Moderate: 32 layers, FFN 9,600. Strong: 28 layers, FFN 8,192.
- Random: uniform contiguous internal depth window, random coupled FFN channels.
- Informed: lowest-NLL contiguous internal depth window on 32 calibration blocks;
  coupled weight L2 magnitude FFN ranking. This is not the pilot's individual-layer
  selection. First and last layers are preserved by both methods.
- Both levels × both criteria × seeds 42/123/2026, KL-only recovery: 12 runs.
- Both levels × informed × same three seeds, CE-only recovery: 6 matched ablations.
- Each run: 1,536 × 128 input tokens = 196,608 tokens, 195,072 causal targets,
  192 updates; rank-8/alpha-16 q/v LoRA; LR 1e-4; microbatch1/accumulation8.
- Full-vocabulary forward KL at T=2, T² scaling, hard CE weight0 for KD.
  CE controls use identical structure, initialization seed, token order and budget.
- Shuffle within each 64-block shard per seed; identical shard order across runs.
  Informed architecture is deterministic; its seeds vary adapters and data order.
- No best-checkpoint/test-based tuning: final fixed-budget checkpoints are reported.
- Before any final-model evaluation, the measurement scope was clarified to also
  time every recovered model at128→16tokens on CUDA (2warmups/5timings), not infer
  a random student's throughput from a same-shaped informed student's timings.
  Three-size paired CPU/GPUdeployment tests remain teacher plus two predeclared
  representative informed seed42students. See amendmentA3.
- Validation recovery at 0/49,152/98,304/196,608 tokens; locked test256 blocks;
  PIQA200 length-normalized conditional likelihood (custom, not official harness).
  Evaluation reuses the prior study's disclosed test partition, not a newly unseen
  external benchmark. Scores must not be interpreted as unbiased after repeated
  project development or absence of pretraining contamination.

## CPU and GPU protocol
1. Actually train/prune/distill a tiny byte-level Qwen transformer on local text,
   in matched FP32 on both CPU and CUDA, several seeds. This is a reduced
   correctness/trend study, not evidence of 3B training equivalence.
2. Run real teacher and two representative recovered students on CPU FP32 and
   GPU FP16, same quality inputs and three generation sizes32/128/512→16.
   This is an explicitly **device-plus-precision deployment comparison**. Tiny
   matched-FP32 cases provide the clean device comparison; never conflate them.
3. Two warmups, five timings, medians and IQR/range, explicit device assertions,
   synchronized CUDA, 4 CPU threads, one benchmark process at a time.
4. Separate loading/construction, tokenization, transfers, prefill+decode,
   request end-to-end and load-inclusive first-request costs. Loading is a fresh
   model load, not guaranteed cold OS file cache. Record phase-separated
   calibration, target generation, pruning/reconstruction, recovery, validation,
   final evaluation, serialization and orchestration costs.
5. Record scientific NLL/PPL and paired PIQA accuracy, hardware/software/driver,
   actual device/dtype, allocator and process RSS plus GPU system samples.
6. Report observed benefit at sampled sizes, not an invented exact crossover.
   RAPIDS/cuML fallback audit: N/A; no cuML path is used.
7. Audit TensorRT-LLM native-Windows, model, GPU and environment support using
   official documentation. Do not install unsupported engines or claim an engine
   benchmark if unavailable. ModelOpt/NeMo are conditional, not required imports.

## Safety and durability
- Preserve FINAL_PROJECT_SUBMISSION and all prior evidence unchanged.
- One GPU model at a time; CUDA allocator limit82%; four CPU compute threads.
- Approximately2.3GiB full-logit shard, deleted only after all18 consumers have
  durable checkpoints. Exact retained recipes + small adapters replace18 full models.
- Export one representative merged model only; extension disk ceiling12GiB,
  keep60GiB free disk and4GiB available RAM. No overclock/driver/system changes.
- GPU ≥80°C: wait for ≤72°C with a bounded cooldown; stop if not recovered.
  GPU temperature monitoring is not a guarantee against power loss or all faults;
  CPU temperature may be unavailable on Windows and is labelled accordingly.
- Atomic replace + fsync checkpoint files; retain prior optimizer checkpoint;
  validate hash on read; checkpoint every optimizer update. At most one update
  per active run is lost to a normal interruption; filesystem/hardware failure
  can still damage data. No uncontrolled automatic reruns on errors.
- run.py is idempotent and resumes only matching configuration/source/data hashes.
  STOP file requests a graceful pause at an update boundary. RESUME.ps1 restarts.

## Portfolio deliverables
Source, configs, focused correctness/resume tests, raw per-block/per-example
results, per-seed summaries/effect sizes, Pareto/recovery/sensitivity/cost figures,
model/data cards, requirement evidence ledger, reproduction commands, technical
report, twelve-slide PPTX/PDF with notes and preview validation.
Publish a lightweight GitHub repository after validation; exclude weights, target
caches, private paths, credentials, environment and copyrighted supplied PDFs.
GitHub authentication must be available; absent authentication is a publishing
blocker, not permission to invent a repository URL.
