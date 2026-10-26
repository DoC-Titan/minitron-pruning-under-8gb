# Third-party attribution and scope

## Model and derivatives
**Built with Qwen.** Qwen2.5-3B and model-derived weights/adapters remain subject
to the Qwen Research License Agreement, copyright Alibaba Cloud. Noncommercial
academic use must comply with those terms. This project does not relicense the
teacher, tokenizer or derivatives as Apache-2.0. Full supplied terms are retained
in `licenses/QWEN_RESEARCH_LICENSE.txt`.

Modifications: physical transformer-layer and coupled FFN-channel removal;
rank-8 LoRA KL/CE recovery; FP16 merged representative export. Exact groups and
training settings are recorded in recipes and config. Base weights/tokenizer
are not uploaded to GitHub by this project.

## Academic anchors
Sreenivas, Muralidharan et al., *LLM Pruning and Distillation in Practice: The
Minitron Approach* (2024), NVIDIA. https://arxiv.org/abs/2408.11796 .
Muralidharan et al., *Compact Language Models via Pruning and Knowledge
Distillation* (2024). https://arxiv.org/abs/2407.14679 .
The study is independently implemented with PyTorch, is not an NVIDIA product,
and does not copy NVIDIA's model results as its own measurements.

The supplied question bank and technical report PDFs retain their original
copyrights and are excluded from publication. Only source citations and the
specific applicable requirements/evidence mapping are provided.

## Data and software
WikiText: Wikipedia-derived text; see its dataset card and original attribution/
share-alike terms. https://huggingface.co/datasets/Salesforce/wikitext .
PIQA: Bisk et al., Physical IQA; dataset source and terms:
https://github.com/ybisk/ybisk.github.io/tree/master/piqa .
PyTorch, Hugging Face Transformers/Datasets/Accelerate/Safetensors, NumPy and
Psutil are separately licensed dependencies. PowerPoint is local proprietary
presentation-rendering software, not redistributed. GitHub CLI is obtained from
the official cli/cli release and its SHA256 is verified before execution; it is
not embedded in the project or installed as a system-wide driver/component.

Project-authored source and research text are supplied for transparent academic
inspection; no blanket license is applied over third-party weights, data, papers
or course material. Consult rights holders before commercial use or redistribution.
