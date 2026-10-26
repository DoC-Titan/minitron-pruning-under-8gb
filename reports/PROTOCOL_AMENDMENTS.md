# Execution amendments / failures

## A1: Windows cache durability handle
After CPU/CUDA tiny-reference completion, before any main3B recovery update,
the first target shard failed at fsync on a read-only handle (Windows EBADF).
Changed `worker.py`: reopen the flushed memmap with `r+b` rather than `rb` before
fsync/rename. No weights, data, objective, seeds, budgets or timed compute changed.
The old source-lock snapshot is retained in state/source_lock.before_A1.json;
the new scientific source lock records the fixed worker hash. Failed-stage wall
time remains in stage costs. The partial owned shard is regenerated on resume.

## Runtime warning disclosure
The tiny reference's512-token prompt plus16generated tokens exceeds its config's
nominal512max positions. Qwen rotary attention executed all cases without an
exception; the actual prompt/output lengths are retained. This is a systems
measurement, not a validated long-context quality claim. No post-hoc timing
selection or removal of that case is performed.

## A2: Completion-receipt recovery at a power-loss boundary
Code review found a narrow interruption window: an optimizer checkpoint could
commit the final update of a shard before its completion JSON was written.
On restart, the worker skipped retraining but did not recreate that receipt.
It now repairs the missing receipt from the verified checkpoint before returning.
A dedicated CPU-only unit test simulates this condition and asserts that no
model reconstruction or retraining occurs. No compute objective, data, RNG,
architecture, budgets or scientific measurement changes. The run may adopt the
fix between subprocesses; old lock is archived as source_lock.before_A2.json.
The checkpoint writer also preserves the last good previous checkpoint instead
of backing up a known-corrupted current file over it; the corruption test now
exercises a second interrupted write after fallback. Scientific code is unchanged
apart from these durable-state handling branches.

## A3: Per-condition GPU throughput coverage
Before any main final-model evaluation, added the same fixed128prompt/16output
CUDA generation benchmark to all18recovered-model evaluations (2warmups,
5synchronized timings, median and measured allocator peaks). This prevents
assuming identical throughput just because random/informed shapes match.
The paired multi-size CPU/GPU design is unchanged. No training objective,
data, seeds, recovery budgets or selection changes; no final results were used
to choose this addition. Worker source lock is updated with its prior snapshot
retained as source_lock.before_A3.json. Raw measurements enter the per-seed CSV.
