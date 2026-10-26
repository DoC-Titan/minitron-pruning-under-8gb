## 1. Smaller models. Stronger evidence.
Open with the engineering constraint, not a claim of reproducing NVIDIA scale. This is built with Qwen and inspired by Minitron. The project contribution is a controlled, auditable compression and systems study.

## 2. Does the least-damaging cut recover best?
Explain that pruning loss and post-recovery quality need not rank architectures identically. Perplexity and downstream task accuracy are different objectives. Negative and metric-dependent results are scientifically useful.

## 3. A controlled experiment—not a model-size demo
There are 12 KL recovery runs and six CE controls. Seeds are 42, 123 and 2026. Both methods use the same architecture within each compression level. This is not a broad hyperparameter sweep.

## 4. Physical pruning with an exact removal recipe
The Qwen gated FFN needs three coupled slices. Residual hidden width remains 2048. The depth procedure is different from the old pilot individual-layer ranking; never mix their provenance.

## 5. Sensitivity is measured before recovery
Walk through one depth-window position and its NLL increase. The accompanying FFN map in the report is a magnitude distribution, not an individually ablated neuron causal-effect map.

## 6. Full-vocabulary KD without two large GPU models
Distinguish teacher targets from hidden states. Full-vocabulary logits retain the exact objective, unlike top-k coarsening. Cached shards are deleted only after all consumers have durable receipts. LoRA is not full-weight Minitron recovery.

## 7. Compression–quality trade-offs, not a universal winner
These are arithmetic means and sample SD over only three seeds. The graph uses mean test NLL; the body uses mean seed perplexity. The finite observed Pareto frontier is descriptive. Do not imply an unmeasured compression-ratio law.

## 8. Recovery and the isolated distillation ablation
Lower NLL is better and higher PIQA is better. A negative KL-minus-CE NLL difference favours KD; the measured positive differences here favour CE on perplexity. The small mean PIQA advantage of KL is not evidence of statistical superiority. The control matches initialization, structure, order, learning rate and tokens. This does not prove a 50-times token-efficiency claim.

## 9. CPU and CUDA both execute—not just a device flag
The reduced model demonstrates algorithmic correctness, not 3B CPU training robustness. The real deployment comparison mixes hardware and precision, so the tiny same-weight FP32 experiment provides a clean device comparison. Byte-model and subword-model perplexities are not comparable.

## 10. Where is GPU deployment practically beneficial?
These are CPU FP32/CUDA FP16 deployment ratios, not pure hardware speedups. Report observed sampled lengths only; when GPU already wins the smallest case, the exact crossover remains unknown below the sampled range. Check the matched-FP32 tiny plot for overhead-dominated cases.

## 11. A recoverable run with a visible resource budget
Allocator peaks are not total board memory. Temperature is sampled, not a continuous hardware assurance; CPU temperature was unavailable. No overclocking or driver changes. A Windows fsync issue was logged and corrected before main training, with no experimental-design change.

## 12. What this project establishes—and what it does not
Close with the actual outcomes from the tables, including any negative results. Qwen2.5-3B uses a noncommercial research license. TensorRT-LLM was conditionally excluded because the cited release requires Linux, while this environment is native Windows. Do not present PyTorch throughput as TensorRT. References: supplied Minitron technical report, question-bank p1/pp20–21, official Qwen and NVIDIA documentation.