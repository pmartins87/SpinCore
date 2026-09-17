# SpinCore Roadmap — active state 2026-09-16

This file tracks the active legacy-first functional training path. Detailed historical R0–R12 engineering snapshots remain preserved in Git history and in the validation/docs tree; they do not silently override the current training plan.

## Active status

- LT0 calibration — **DONE**: 120k roots.
- LT1 production-shaped milestone — **DONE**: 1.2M roots.
- LT1 physical fit optimization — **PASS**: 31 root workers, 8 parent Torch threads, vectorized batching.
- LT2 Stage A — **PASS**: iteration 3000 / 1.8M roots total.
- LT2 Stage A resource gate — **PASS**: no swap, min WSL MemAvailable about 10.67 GiB, final checkpoint 2.236 GiB.
- LT2 Stage A first policy-reservoir saturation transition — **PASS**: 3H AveragePolicy crossed 2M; HU remains at 820,667.
- Weak-baseline learning curve — **POSITIVE OVERALL/3H; HU STILL NOISY/FLAT**.
- Immediate gate — **one bounded read-only concurrent-fit benchmark**.
- LT2 Stage B — **PENDING fit-screen result**.
- DeepCrusher faithful oracle — **build in parallel**.

Canonical current files:

- `CURRENT_WORK.md`
- `docs/LONG_TRAINING_PLAN.md`
- `docs/LT2_STAGE_A_REVIEW_20260916.md`
- `docs/LT1_COMPLETION_REVIEW_20260916.md`
- `docs/LT1_FIT_BENCHMARK_RESULT_20260916.md`

## Learning milestones

### LT0 — calibration

Purpose: prove the repaired pipeline trains, saves, resumes and plays complete 3H/HU hands. It is not a final-strength run.

### LT1 — production-shaped scale milestone

Purpose: establish large reservoirs, realistic sampler pressure, checkpoint cost and a measured Ryzen execution profile.

Result: 2000 iterations / 1.2M roots completed and preserved.

### LT2 Stage A — first saturation gate

Purpose: continue the exact LT1 state until the first policy reservoir reaches the 2M cap, then review memory/checkpoint behavior before longer runs.

Result: 3000 iterations / 1.8M roots completed. 3H AveragePolicy crossed 2M; HU is 820,667. No swap or runaway checkpoint growth. See `docs/LT2_STAGE_A_REVIEW_20260916.md`.

Weak-baseline learning trend from 120k -> 1.2M -> 1.8M:

- uniform overall cEV: +10.836 -> +15.873 -> +17.015;
- uniform 3H cEV: +2.407 -> +8.898 -> +12.769;
- passive-caller overall cEV: -2.402 -> -0.553 -> -0.064;
- jammer overall cEV: -5.498 -> -2.719 -> -1.298;
- jammer 3H cEV: +2.295 -> +4.045 -> +7.422.

HU did not improve from LT1 to LT2-A on the fixed-seed point estimates. That remains a tracked warning, not a stop signal, because HU intervals are wide and its policy reservoir is still far below capacity.

## Immediate execution gate

Run `tools/benchmark_lean_lt2_concurrent_fit.sh` exactly once.

Purpose: determine whether overlapping the independent 3H/HU Advantage optimizer loops can recover meaningful Ryzen throughput without changing fit results.

Acceptance to justify full-iteration integration:

- exact same-thread sequential/concurrent model hashes;
- exact final losses;
- exact per-domain batch RNG states;
- source checkpoint unchanged;
- >=5% median fit-wall improvement versus sequential 8-thread fitting.

If this screen fails, stop tuning and retain 31 workers / 8 threads / vectorized batches. If it passes, build one disposable full-iteration candidate and prove exact learning-state parity before using concurrency in production.

## LT2 Stage B

Once the execution path is cleared, continue the same iteration-3000 LT2 checkpoint to approximately iteration **7500**:

- +4500 iterations;
- +2.7M roots;
- 4.5M roots total.

Why 7500: the Stage-A HU AveragePolicy sample rate projects the 2M HU reservoir crossing near iteration 7.3k. This makes ~7500 the next natural bounded training milestone where all four 2M reservoirs should effectively be in saturation/replacement regime.

At Stage B completion, stop and review:

- memory/swap with all large reservoirs saturated;
- checkpoint size/save time;
- throughput;
- 3H learning continuation;
- HU learning trend;
- weak-opponent regressions;
- DeepCrusher head-to-head if the faithful oracle is ready.

## Strength tracking

Weak fixed opponents are regression sentinels, not the final target. Product acceptance ultimately requires a faithful, balanced SpinCore-vs-DeepCrusher benchmark under common game semantics, followed later by full tournament progression once that simulator is frozen.

Do not use a simplified DeepCrusher imitation for canonical claims.

## Persistent rules

- Continue the same LT1/LT2 learning state; do not restart without concrete evidence.
- Preserve LT0, LT1 and milestone LT2 checkpoints.
- Do not shrink 2M reservoirs just to make infrastructure easier.
- CPU utilization is telemetry, not a target by itself; only adopt optimizations that improve wall time while preserving learning semantics.
- Do not rerun closed tuning matrices absent a materially changed workload.
- Do not judge architecture ceiling from early weak-baseline results.
- Stop larger training blocks at meaningful evidence checkpoints rather than training indefinitely without measurement.
