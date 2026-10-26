# Model card: physically pruned Qwen2.5-3B research students

**Built with Qwen.** Base: `Qwen/Qwen2.5-3B`, revision
`3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`. Qwen Research License,
noncommercial academic use subject to its terms—not Apache-2.0.

## Intended use
College research into structured compression, distillation and local execution.
Use as a causal language-model completion demo, not a safety-aligned chatbot,
medical/legal tool or production decision system.

## Modifications
- Moderate: 36→32 transformer blocks and FFN11,008→9,600.
- Strong: 36→28 blocks and FFN11,008→8,192.
- Physical removal, not sparse masking. Attention head counts, hidden/residual
  dimension2,048, vocabulary151,936 and tokenizer remain unchanged.
- Random contiguous-window/channel baseline versus calibration-window NLL and
  coupled FFN magnitude selection. Exact IDs: `recipes/`.
- Rank8/alpha16 q/v LoRA, LR1e-4, no dropout, 196,608 input tokens each;
  KL-only or CE-only matched recovery. Final adapters are FP32; merged inference
  weights are FP16. CPU tests load the same rounded weights as FP32.
- The single exported checkpoint is **strong/informed/seed42/KL**, a predeclared
  representative, not a post-hoc best-of-test model.

## Evidence
`reports/PER_SEED_RESULTS.csv`, `reports/AGGREGATE_RESULTS.csv`, raw final metrics,
`reports/VALIDATION.json`, CPU/GPU benchmark receipts and phase costs. These files
exist as completed findings only after their validation stages finish.

## Limits
Short-context WikiText and custom PIQA200 are narrow quality probes. Generation
can be incorrect, biased or harmful; structural pruning can further degrade
reasoning and robustness. No instruction tuning, teacher correction or broad
benchmark claims. Only three seeds and one machine/family. Existing test partition
is disclosed as reused from earlier project development. Training tokens are tiny
relative to NVIDIA Minitron's billions. LoRA-only main recovery is not full-weight
retraining. Quantization was not performed or counted as parameter reduction.

## Distribution
Large teacher/student weights are not included in GitHub. Obtain the pinned base
subject to its license; reconstruct with the saved recipe and adapter. Preserve
license/attribution/modification notices when sharing derivatives. Adapters are
Qwen-derived research artifacts, not a relicensing of the base. Full notices are
in `THIRD_PARTY_NOTICES.md` and `licenses/QWEN_RESEARCH_LICENSE.txt`.
