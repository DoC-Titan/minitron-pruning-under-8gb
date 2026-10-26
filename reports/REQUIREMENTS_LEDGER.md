# Requirement-to-evidence ledger

| Requirement | Executed evidence / artifact | Scope |
|---|---|---|
| Open1B–8Bmodel/licensing/VRAM | HARDWARE_SOFTWARE.json, MODEL_CARD.md, preflight receipt |3Bnoncommercial research |
| Two physical dimensions, exact groups | recipes/*.json, unit shape/save/reload test | Depth + coupled FFN |
| Random versus sensitivity/magnitude | PER_SEED_RESULTS.csv, sensitivity.json | Matched architectures within each level |
| Teacher distillation/recovery versus tokens | Full-vocab cache manifests, per-run recovery logs, recovery.svg |196608tokens each; LoRA |
| PPL/task/parameters/memory/throughput | Raw final metrics, CPU_GPU*.csv, phase costs | Custom PIQA and short-context PPL |
| Immediate damage versus final ranking | PAIRED_EFFECTS.json and initial/final NLL columns | Metric/level/seed-specific |
| Executed reduced CPU prune+distill | tiny_cpu.json, TINY_CPU_GPU_RESULTS.csv | Tiny byte LM, not3BCPUtraining |
| Actual NVIDIA execution/fallback audit | Parameter-device assertions, CUDA timings, HARDWARE_SOFTWARE.json | No auto fallback; cuML N/A |
| Multi-size warm repeated median timings | benchmark*.json, tiny benchmark JSON |2warmups/5timings,3real sizes/5tiny sizes |
| CPU/GPU scientific comparison | CPU_GPU_QUALITY.csv; matchedFP32tinyquality | Deployment precision confound explicitly labelled |
| Several seeds | Three seeds in18mainruns and CPU/CUDA tiny runs | Seed42timing representative, not replicated hardware sessions |
| Phase-separated costs, complete metadata | PHASE_COSTS.json, raw benchmarks, HARDWARE_SOFTWARE.json | Checkpoint counter approximate; no CPU temperature |
| Observed GPU practical benefit scale | CPU_GPU_DEPLOYMENT.csv, TINY_MATCHED_FP32_SCALING.csv | Sampled benefit only; no exact unsampled crossover |
| Pareto / sensitivity / distillation ablation | figures/*.svg, PARETO_POINTS.json, PAIRED_EFFECTS.json | KL-only versus CE-only matched control |
| TensorRT-LLM if supported | BACKEND_SUPPORT_AUDIT.md | Native-Windows cited release unsupported; not executed |
| Durable execution and reproducibility | Atomic hashed checkpoints, corruption-fallback unit test, RESUME.ps1 | No UPS or hardware-fault guarantee |
| New12-slide presentation | presentation/ after build validation | Generated strictly from measured summary |
| GitHub publication | publish_receipt.local.json only after verified remote | Pending authentication/publication until receipt exists |