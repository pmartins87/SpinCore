# SpinCore Roadmap — active state 2026-09-17

This file tracks the active legacy-first functional training path. Historical engineering snapshots remain preserved in Git history and validation/docs; they do not override the current training plan.

## Active status

- LT0 calibration — **DONE**: 120k roots.
- LT1 production-shaped milestone — **DONE**: 1.2M roots.
- LT1 physical fit optimization — **PASS**: 31 root workers, 8 parent Torch threads, vectorized batching.
- LT2 Stage A — **PASS**: iteration 3000 / 1.8M roots total.
- LT2 Stage A resource gate — **PASS**: no swap, min WSL MemAvailable about 10.67 GiB, final checkpoint 2.236 GiB.
- LT2 first policy-reservoir saturation transition — **PASS**: 3H AveragePolicy crossed 2M; HU ended at 820,667.
- Weak-baseline learning curve — **POSITIVE OVERALL/3H; HU STILL NOISY/FLAT**.
- Concurrent-fit microbenchmark — **PASS**: 1.312188x fit speedup with exact same-thread fit parity.
- Concurrent-fit full-iteration concept parity — **PASS**.
- Concurrent-fit production-function parity — **PASS**: semantic parity exact, source unchanged, first_difference null.
- LT2 Stage B — **READY**: iteration 3000 -> 7500 / 4.5M roots total.
- DeepCrusher faithful oracle — **BUILD IN PARALLEL**.

Canonical current files:

- `CURRENT_WORK.md`
- `docs/LONG_TRAINING_PLAN.md`
- `docs/LT2_STAGE_A_REVIEW_20260916.md`
- `docs/LT2_PRODUCTION_CONCURRENT_PARITY_RESULT_20260917.md`

## Learning milestones

### LT0 — calibration

Purpose: prove the repaired pipeline trains, saves, resumes and plays complete 3H/HU hands. It is not a final-strength run.

### LT1 — production-shaped scale milestone

Purpose: establish large reservoirs, realistic sampler pressure, checkpoint cost and a measured Ryzen execution profile.

Result: 2000 iterations / 1.2M roots completed and preserved.

### LT2 Stage A — first saturation gate

Purpose: continue the exact LT1 state until the first policy reservoir reaches the 2M cap, then review memory/checkpoint behavior before longer runs.

Result: 3000 iterations / 1.8M roots completed. 3H AveragePolicy crossed 2M; HU is 820,667. No swap or runaway checkpoint growth.

Weak-baseline learning trend from 120k -> 1.2M -> 1.8M:

- uniform overall cEV: +10.836 -> +15.873 -> +17.015;
- uniform 3H cEV: +2.407 -> +8.898 -> +12.769;
- passive-caller overall cEV: -2.402 -> -0.553 -> -0.064;
- jammer overall cEV: -5.498 -> -2.719 -> -1.298;
- jammer 3H cEV: +2.295 -> +4.045 -> +7.422.

HU did not improve from LT1 to LT2-A on the fixed-seed point estimates. That remains a tracked warning, not a stop signal, because HU intervals are wide and its policy reservoir is still far below capacity.

## Execution optimization — CLOSED

The fit-concurrency screen is complete and admitted.

Repeated 8-thread result:

- sequential combined fit median 7.1461 s;
- concurrent fit median 5.4460 s;
- speedup 1.312188x;
- about 23.79% fit-wall reduction;
- exact same-thread model/loss/RNG parity.

The production iteration function was then validated against canonical sequential iteration 3001 and passed exact semantic parity for models, optimizers, sampler/domain RNGs, counters, reservoir RNGs, added sample streams and non-timing reports. `first_difference=null` and the source checkpoint remained unchanged.

No further fit tuning before Stage B unless workload materially changes.

## LT2 Stage B — ACTIVE NEXT MILESTONE

Canonical launcher:

`tools/run_long_training_lt2_stage_b.sh`

Contract:

- source: preserved iteration-3000 LT2 Stage A checkpoint;
- source SHA256: `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`;
- target iteration: **7500**;
- +4500 iterations / +2.7M roots;
- 4.5M roots total;
- 31 root workers;
- 8 parent Torch threads;
- vectorized batch construction;
- production `concurrent_fit` iteration mode;
- checkpoint every 250 iterations;
- one-minute memory/swap telemetry;
- isolated run directory; Stage A source remains read-only.

Why 7500: the Stage-A HU AveragePolicy sample rate projects its 2M reservoir crossing near iteration 7.3k, so 7500 is the next bounded point to inspect the all-four-reservoir saturated/replacement regime.

At Stage B completion, stop and review:

- whether HU AveragePolicy crossed 2M;
- RAM/swap with all reservoirs saturated or near saturated;
- checkpoint size/save-time behavior;
- actual throughput under production concurrent fit;
- 3H learning continuation;
- HU learning continuation/regression;
- direct DeepCrusher evidence if the faithful oracle is available.

Do not auto-extend beyond 7500.

## Later LT2 / LT3

If Stage B remains healthy and strategic improvement continues, continue the same learning state through larger resumable blocks measured in millions and eventually tens/hundreds of millions of roots.

Do not choose the final root count by calendar time alone. Continue while meaningful improvement remains measurable and semantics/resources remain healthy.

## Product strength path

Weak fixed opponents remain regression/learning sentinels, not the final target.

Future product evidence must include:

- faithful DeepCrusher R8 v22 direct paired chip-EV;
- HU and 3H separately;
- stack/blind/position breakdowns;
- later full Spin & Go tournament win rate after continuous tournament progression is frozen.

The DeepCrusher oracle must reproduce the frozen OpenPPL strategy faithfully; do not substitute a simplified imitation for canonical claims.

## Immediate action

Run Stage B only:

```bash
bash tools/run_long_training_lt2_stage_b.sh
```

Wait until the launcher prints `LT2_STAGE_B_PASS` or `LT2_STAGE_B_FAIL`. On PASS, stop and review evidence before more training. On FAIL, do not restart automatically.
