# Optional NVIDIA tooling compatibility audit

Checked 26 October 2026 against the installed native-Windows environment.

| Component | Decision | Evidence |
|---|---|---|
| CUDA-enabled PyTorch | Core NVIDIA execution path | Installed2.5.1+cu121; actual model/update tests and receipts |
| TensorRT-LLM | **Not executed: current native-Windows stack unsupported by cited Linux release** | Official support matrix requires Linux x86_64/aarch64; environment is Windows |
| RTX4060/Ada | Architecture family appears in cited support matrix | This does not guarantee every precision, model or customized architecture |
| Qwen2 family | Listed in model matrix | Pruned shapes still require engine conversion and validation; not automatically validated |
| Model Optimizer/NeMo | Not used | Custom physical PyTorch slicing and standard HF serialization avoid additional stack changes |
| RAPIDS/cuML/cuml.accel | Not applicable | No classical estimator or cuML acceleration/fallback path |

Sources accessed:
- https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html
  (page reports last updated15September2025, commit0c9430e).
  Exact OS statement: “TensorRT-LLM requires Linux x86_64 or Linux aarch64.”
- https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html
  (page reports last updated15May2026, commit11e16cd).
  Describes Linux/Docker installation; approximately63GB to build a container.

These are release-specific observations, not a claim that every historical
TensorRT-LLM version or experimental Windows port is impossible. No WSL/Docker/
driver installation or63GB engine build is justified within this storage-safe
extension. The rubric's “if supported” condition is recorded transparently.
All throughput in this project is PyTorch SDPA, **not TensorRT-LLM performance**.

The HF Qwen configuration disables sliding-window attention. Transformers emits
an SDPA sliding-window warning from the non-null configuration field; this study
uses ordinary causal full attention at short contexts, not a supported sliding-
window-kernel claim. Dropout is zero. Execution dtype is recorded explicitly,
regardless of the checkpoint config's historical torch_dtype field.
