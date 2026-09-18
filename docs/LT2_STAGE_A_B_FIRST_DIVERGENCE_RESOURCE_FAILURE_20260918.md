# SpinCore — LT2 Stage-A -> Stage-B forensic first-run resource termination

Date: 2026-09-18
Status: **DIAGNOSED — WORKER MEMORY DESIGN FIXED; FORENSIC SEMANTICS UNCHANGED**

## Observed failure

The first full first-divergence forensic was launched with:

- six already-seen forensic seeds;
- 5,000 scenarios per seed;
- 31 spawned workers.

After the header, the process exited with only:

`Terminated`

followed by a multiprocessing `resource_tracker` warning about leaked semaphores.

There was no Python traceback and no report.

This is consistent with abrupt OS/WSL process termination, most plausibly memory pressure.

## Concrete memory amplification in the first implementation

Each spawned worker called `LeanFunctionalAgent.from_checkpoint` twice:

- once for Stage A;
- once for Stage B.

That loader performs `torch.load` on the **entire training checkpoint** before selecting policy weights.

The training checkpoints contain much more than the deployed AveragePolicy:

- Advantage model/optimizer;
- AveragePolicy model/optimizer;
- large Advantage reservoir;
- large strategy reservoir;
- both THREE_HANDED and TRUE_HEADS_UP domains;
- history/configuration state.

The forensic uses only the TRUE_HEADS_UP AveragePolicy.

Therefore 31 spawned workers were redundantly deserializing two complete training checkpoints each. Even if RAM exhaustion was not independently confirmed by an OOM log, this was an unnecessary and unsafe resource design.

The semaphore warning is treated as a consequence of abrupt multiprocessing shutdown, not as the root cause.

## Fix

The forensic now extracts lightweight HU AveragePolicy snapshots **once in the parent process**:

- source checkpoint is read once;
- mmap loading is used when supported;
- only `domains["TRUE_HEADS_UP"]["policy"]` is copied;
- a tiny derived policy-only snapshot is written in the forensic run directory;
- full checkpoint/reservoir objects are released before the worker pool starts.

Spawned workers now load only:

- the solver library;
- Stage-A HU AveragePolicy snapshot;
- Stage-B HU AveragePolicy snapshot.

They never deserialize the training reservoirs.

Default worker count is reduced from `31` to `16`.

This changes only resource handling. It does **not** change:

- scenario/deal sampling;
- seed family;
- Stage-A/Stage-B policy weights;
- action semantics;
- pairing;
- first-divergence attribution;
- statistical summaries.

## Checkpoint safety

Stage A and Stage B remain read-only and are hash-checked after the audit.

No K4 training is authorized.

## Next action

Rerun:

```bash
bash tools/run_lt2_stage_a_b_first_divergence.sh
```

Expected additional startup output includes lightweight policy snapshot sizes and `workers=16`.

Stop on the first error or on:

`LT2_STAGE_A_B_FIRST_DIVERGENCE_FORENSIC_PASS`.
