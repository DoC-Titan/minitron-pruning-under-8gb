# Portfolio walkthrough and viva preparation

## Five-minute story
1. **Constraint:** compress an already trained, open 1B–8B model on an 8 GiB GPU;
   preserve evidence while controlling VRAM, disk use and interruption risk.
2. **Question:** does a calibration criterion that minimizes immediate loss also
   select the best equal-budget recovered model? Check size and metric dependence.
3. **Method:** physically remove blocks and coupled gated-FFN channels, compare
   random and informed structures, recover with genuine full-vocabulary teacher KL.
4. **Controls:** matched architectures/tokens/LoRA settings; three independent
   pruning/training seeds; objective-only CE controls; fixed evaluation protocol.
5. **Systems:** actual CPU and CUDA paths, warm repeated timings, scientific
   quality comparison, complete metadata and separated workflow costs.
6. **Conclusion:** read the measured effects table rather than memorizing an
   assumed 'informed is always better' claim. Report negative/metric-dependent
   results honestly.

## Offline demonstration
Use the retained local `model/` export after `EXPORT_COMPLETE.json` and validation
exist. Run `demo.py --device cuda` for a short bounded generation, or `--device cpu`
for the explicitly labelled FP32 CPU path. Neither downloads or starts training.
The GitHub repository intentionally omits multi-gigabyte weights; reconstruct
with `reconstruct.py` from the pinned local teacher, recipe and final adapter.
Keep generated text separate from quantitative evaluation; one plausible response
does not establish language-model quality or safety.

## Questions you must be able to answer
- **Is this actual compression?** Yes: tensor shapes and block counts change,
  parameter totals decrease, and a standard HF checkpoint reloads. It is not masking.
- **Why three FFN slices?** In a gated FFN, each intermediate channel couples a
  gate row, up row and down-projection column. Breaking that coupling breaks semantics.
- **Why preserve residual width?** Keeping it fixed avoids changing every residual,
  attention, norm and embedding interface, making two-axis pruning tractable here.
- **What is the KL direction?** Forward KL(teacher‖student), with both distributions
  softened at T=2 and loss multiplied by T²; full vocabulary, no hard-label term
  in the KL condition. The CE condition is a separate controlled recovery baseline.
- **Why not keep teacher and student on the GPU?** Their weights and training
  state compete for 8 GiB. Offline full-logit shards make residency sequential.
- **Why not store all logits?** One prior cache was approximately55GiB. Streaming
  a64-block shard bounds live target storage near2.3GiB without changing the objective.
- **Does mixed precision count as parameter compression?** No. Physical parameter
  removal, numeric precision and serialized file size are reported separately.
- **Are timing repeats seeds?** No. Five timing repeats measure local timing
  variability; seeds42/123/2026 vary stochastic pruning/training conditions.
- **Why a CPU tiny model as well as real-model inference?** It provides an executed
  complete no-GPU algorithmic path and clean matched-FP32 device experiment. It
  does not prove multi-billion-model CPU training is practical or equivalent.
- **Can the real CPU/GPU ratio be called pure GPU speedup?** No: CPU FP32 versus
  CUDA FP16 mixes precision and hardware. The report labels it a deployment ratio.
- **What is the crossover scale?** Only observed sampled sizes can be identified.
  A win at every tested size means the exact threshold below that range is unknown.
- **Why no TensorRT-LLM number?** The cited supported release requires Linux,
  while the workstation is native Windows. The conditional support audit is
  explicit; PyTorch throughput is not renamed TensorRT throughput.
- **How close is this to NVIDIA Minitron?** Shared structural/KL principles, but
  main recovery is short-budget LoRA, not94B–380B full-weight training, teacher
  correction, architecture search or alignment. We do not inherit NVIDIA's scores.
- **How do checkpoints survive interruptions?** Write temporary files, flush,
  atomically replace, verify hashes, retain a prior valid state, and record shard
  consumption before cache reclamation. Tests simulate corruption and a missing
  completion receipt. Hardware/filesystem failure can still lose data.
- **What is not controlled?** Desktop load, cold OS cache, broad pretraining
  contamination, semantic overlap and cross-family generalization. Three seeds
  and custom PIQA200 do not justify universal or statistically definitive rankings.
- **May I use this commercially?** Do not assume so. Qwen2.5-3B uses the Qwen
  Research License; preserve notices and obtain appropriate permission.

## Presentation checklist
Open the twelve-slide PDF as a fallback to PowerPoint. Use the report tables for
precise figures and speaker notes for caveats. Demonstrate only a short known
prompt. Have `reports/REQUIREMENTS_LEDGER.md`, the source/config/recipe, and raw
benchmark receipts ready for methodological questions. Keep the private GitHub
repository visible to invited reviewers; do not expose auth tokens or local paths.
