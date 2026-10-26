# Data card

## Main subword study
- WikiText-2 raw, `Salesforce/wikitext`, pinned dataset revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`.
- Reuses original article-partitioned calibration/training and disclosed
  validation/test tensors; hashes and provenance: `data/manifest.json`.
- Train1,536 full blocks ×128tokens; no padding. Each block scores127causal
  targets. Incomplete source article tails were dropped by original preparation.
- Calibration32blocks; validation64blocks; test256blocks. Exact complete-block
  overlaps across these groups are checked. Original manifest records exact
  source article checks. No semantic, near-duplicate or pretraining-contamination
  guarantee is implied.
- PIQA200 from its validation split, fixed prior-project selection. Prompt:
  `Question: {goal}\nAnswer:`; separate tokenized space-prefixed alternatives;
  mean conditional continuation log-likelihood; no official lm-eval score claim.
- CPU/GPU scientific comparison uses the same first16testblocks/32PIQAexamples;
  these small paired subsets are not replacements for the main final evaluation.

## Reduced byte reference
UTF-8 byte vocabulary256, fixed before training. Local WikiText official train,
validation and test splits; prefixes262,144/32,768/32,768bytes. Tiny teacher is
trained from scratch, then physically pruned and recovered; all inputs/indices
are deterministic from declared seeds. Same teacher/student design on CPU/CUDA.
Byte PPL must not be compared numerically with Qwen subword PPL.

## Data rights and public repository
WikiText includes Wikipedia-derived material under its dataset terms and
Wikipedia attribution/share-alike conditions. PIQA retains its source terms.
Source references: https://huggingface.co/datasets/Salesforce/wikitext and
https://github.com/ybisk/ybisk.github.io/tree/master/piqa .
Do not claim ownership of source text. Publication includes only bounded research
subsets/provenance needed to reproduce the experiment, with notices; no supplied
course/question-bank or NVIDIA report PDF is republished. Benchmark completion
samples and per-example task scores are research outputs, not new ground truth.
