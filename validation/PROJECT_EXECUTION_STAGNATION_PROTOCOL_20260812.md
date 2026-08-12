# SpinCore execution stagnation protocol — 2026-08-12

This project must surface loss of forward progress immediately instead of hiding it behind generic `in progress` status.

## Status classes

### `STAGNATED`

Use when a process is no longer making credible forward progress, for example:

- runtime materially exceeds the relevant historical/benchmarked envelope with no explanatory workload increase;
- heartbeat/telemetry shows repeated no-CPU/no-child activity while the job remains alive;
- the same failure/retry cycle repeats without new evidence;
- a workflow is waiting because of an implementation/observability defect that can be actively corrected.

When `STAGNATED`, the next project update must state that **at the beginning** and immediately investigate/correct the cause. Waiting without diagnosis is not an acceptable next step.

### `LONG_RUNNING`

Use when computation is slow but remains credible relative to historical scale/workload evidence. A long-running calculation is not automatically stagnated.

When `LONG_RUNNING`:

- report the relevant elapsed-time baseline when available;
- preserve already-spent valid compute rather than cancelling reflexively;
- add/consult heartbeat observability for future/current jobs where possible;
- continue independent safe engineering work while the job computes;
- never infer PASS from elapsed time or apparent liveness.

### `BLOCKED`

Use when a required gate/evidence/dependency is absent or has failed, so downstream execution is unauthorized.

When `BLOCKED`, state that at the beginning and work on the blocking dependency or another explicitly safe prerequisite. Never bypass the gate.

## Non-negotiable rules

Runtime pressure must never be solved by changing strategic gates, tolerances, held-out seeds, accepted sample counts, frozen RNG semantics, or strategy parameters merely to finish faster.

Observability changes must be semantically inert: they may inspect process state and report progress, but must not consume application RNG, mutate training state, alter command arguments, or change child exit semantics.

`READY FOR TABLES` remains forbidden before R12 regardless of execution status.
