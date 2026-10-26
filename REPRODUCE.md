# Reproduce, inspect and recover

## Inspect without running training
Read `reports/TECHNICAL_REPORT.md`, CSV tables, raw `metrics/`, `figures/`,
`EXPERIMENT_PLAN.md`, and `presentation/`. No GPU is needed to inspect evidence.
The original sealed `FINAL_PROJECT_SUBMISSION` remains independent and unchanged.

## Current workstation
From `D:\AIML MiniProject`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\LEVEL4_PORTFOLIO\RESUME.ps1
```

This resumes matching durable checkpoints and skips completed stages. Never
launch two runners. The driver takes a nonblocking file lock. After a power cut,
restore power, inspect `RUN_STATUS.json` and `logs/runner.log`, then run this same
command. No automatic startup task or OS power setting was installed.

To pause safely, create an empty `LEVEL4_PORTFOLIO/STOP` file. The active recovery
worker checkpoints at its next update boundary, then exits. Remove only this
STOP file when ready to resume. Do not delete or edit checkpoint files to force
a rerun. A normal interruption loses at most the uncommitted update for that job;
disk/OS/hardware failures can still require recovery from the previous checkpoint.

## Environment and portability
Recorded versions and hardware: `reports/HARDWARE_SOFTWARE.json`. Working setup:
Python3.11.9, PyTorch2.5.1+cu121, Transformers4.51.3, NumPy, Accelerate,
Safetensors, Datasets and Psutil. Presentation build uses installed PowerPoint16
COM on Windows; science and SVG/CSV report generation do not require PowerPoint.
No RAPIDS, cuML, TensorRT-LLM or NeMo dependency is required by this implementation.

Set `LEVEL4_TEACHER` to a local pinned Hugging Face teacher snapshot if its
default relative path is unavailable. Runtime loading is strictly local-only;
there are no automatic model downloads. The initial workstation `prepare` stage
reuses preserved original data artifacts; public repository includes its locked
bounded prepared tensors and source manifest so downstream stages do not need the
original workspace. A fresh checkout must not pretend the old execution state is
present: preserve its measured metrics, and use the independent-run bootstrap:

```powershell
python replicate.py --output ..\independent_replication --teacher C:\path\to\pinned_local_Qwen
# Preparation only: verifies the teacher/input hashes; copies no old metrics or checkpoints.
# Start hours-long replication explicitly when ready:
powershell -NoProfile -ExecutionPolicy Bypass -File ..\independent_replication\START_REPLICA.local.ps1
```

`replicate.py --run` combines those two steps. Existing output directories are
protected. The native-Windows orchestrator lock uses msvcrt; direct
scientific worker primitives are device-explicit PyTorch.

```powershell
# Focused mathematical, physical-shape and resume-equivalence checks:
python tests.py
# Reduced complete CPU path, with prepared byte corpus:
python tiny_reference.py --device cpu
# Fresh CUDA run requires enough available VRAM and the declared guards:
python run.py
```

Never combine results from modified configs/source with the existing run. The
source lock rejects changes; documented I/O-only amendments are archived rather
than silently mixed. Adapters and recipes can be reconstructed from the local
teacher using `reconstruct.py`, without redoing distillation:

```powershell
python reconstruct.py --job strong_informed_s42_kl --device cuda --output model_rebuilt
```

The command refuses existing output folders. Main scientific outputs use a
specific merge: FP16 base plus FP16-rounded FP32 LoRA delta. Re-evaluate if changing
precision, kernels or library versions. Reproduction is not a bitwise cross-backend
guarantee.

## Storage / memory
One2.3GiB teacher-target payload at a time; exact full-vocabulary supervision.
Only after all consumer checkpoints have durable receipts is that owned payload
removed. All18final adapters and recipes remain; only one full merged model is
exported.12GiB extension limit,60GiB free-disk reserve,4GiB available-RAM reserve,
82% CUDA allocator limit,4 CPU threads,80°C GPU pause/72°C resume. Allocator limits
do not reserve VRAM against other applications; close GPU-heavy applications if a
safety/OOM stop occurs. Do not disable hardware thermal protection.

## Publishing
After completed validation and deck generation, `publish.py` audits an explicit
allowlist, size and secret/path checks, then creates a **private** GitHub repository
by default. It requires authenticated GitHub CLI; never paste tokens/passwords
into reports or chat. A public repository requires the explicit `--public` flag.
Models, targets, optimizer checkpoints, environment directories and supplied PDFs
are excluded. An actual verified remote URL is written only after a successful
push; local preparation is not called publication.

An optional manual-only GitHub Actions CPU correctness **template** is provided
at `examples/cpu-correctness-workflow.yml`, not installed as an active workflow.
Copy it into `.github/workflows/` only if you choose to enable it with appropriate
GitHub permissions. It has not been executed by this local study. Starting it uses
your GitHub Actions quota; check account limits first. Local tests are the executed
evidence, not an unrun CI badge. Publishing this portfolio does not request the
additional OAuth `workflow` scope merely to upload an optional template.
