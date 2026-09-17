# SpinCore — Long-Training Plan

Status: **CANONICAL TRAINING DIRECTION — LT2 STAGE B READY**
Date: 2026-09-17

## Current state

The active learning line is continuous:

- LT0: 120k roots — calibration only;
- LT1: 1.2M roots — production-shaped scale milestone;
- LT2 Stage A: 1.8M roots total — resource/saturation gate PASS;
- concurrent-fit production path — exact semantic parity PASS;
- LT2 Stage B: target iteration 7500 / 4.5M roots total — READY.

Read `LT2_STAGE_A_REVIEW_20260916.md` and `LT2_PRODUCTION_CONCURRENT_PARITY_RESULT_20260917.md` for current evidence.

## Core training contract

The functional line uses:

- empirical SpinGo 3H/HU/blind/stack sampling;
- WTA chip-EV utility scaled by 1500;
- SPNNIV1 frozen-control representation for the current line;
- mature legacy action vocabulary;
- external-sampling Deep CFR with repaired all-nonpositive regret fallback;
- separate 3H and HU brains;
- sampled AveragePolicy trajectories;
- 2,000,000-sample reservoir capacity per memory per domain;
- 600 roots per iteration;
- 100 Advantage optimizer steps per domain per iteration;
- batch size 1024;
- 4000 AveragePolicy optimizer steps per domain at milestone finalization.

Current admitted Ryzen execution profile:

- 31 root workers;
- one numerical-library thread per root worker;
- 8 parent Torch threads;
- vectorized neural batch construction;
- production `concurrent_fit` iteration mode, overlapping only independent 3H/HU Advantage optimizer loops.

The canonical sequential path remains preserved as the default trainer mode for regression/portability.

## LT0 — calibration — DONE

120k roots proved the repaired pipeline trains, saves, resumes and plays complete 3H/HU hands. It is not a competitive-strength checkpoint.

Preserve `runs/lean_first_training/20260915_131133/checkpoint.pt`.

## LT1 — production-shaped milestone — DONE

LT1 completed 2000 iterations / 1.2M roots and finalization successfully.

Preserve `/home/rz9/spincore_lean_functional/runs/long_training_lt1/20260915_181249/checkpoint.pt`.

SHA256: `beef9bee9439de8d9153450d190e62b0388a678b0778c4185108361f626b4337`.

LT1 established the 2M reservoir line and exposed neural Advantage fitting as the dominant measured phase. The physical fit optimization selected 8 parent Torch threads plus vectorized batching. Do not rerun the old 1/2/4/8/16-thread matrix absent a materially changed workload.

## LT2 Stage A — first policy-reservoir saturation gate — PASS

Stage A continued the exact LT1 state through iterations 2001–3000, adding 600k roots and reaching 1.8M total.

Preserve `/home/rz9/spincore_lean_functional/runs/long_training_lt2/20260916_015559/checkpoint.pt`.

SHA256: `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Resource result:

- wall scope about 2h55m47s;
- min WSL MemAvailable about 10.67 GiB;
- swap used 0;
- process max RSS about 16.3 GiB;
- final checkpoint 2.236 GiB;
- finalized save about 54.5 s.

Reservoir result:

- both Advantage reservoirs saturated;
- 3H AveragePolicy crossed 2M and is in replacement regime;
- HU AveragePolicy ended at 820,667 and remained below capacity.

Checkpoint growth slowed after 3H policy saturation, which is expected healthy behavior rather than runaway serialization growth.

## Learning evidence after Stage A

Fixed-seed weak-baseline trend from 120k -> 1.2M -> 1.8M roots:

- uniform overall cEV: +10.836 -> +15.873 -> +17.015;
- uniform 3H cEV: +2.407 -> +8.898 -> +12.769;
- passive-caller overall cEV: -2.402 -> -0.553 -> -0.064;
- jammer overall cEV: -5.498 -> -2.719 -> -1.298;
- jammer 3H cEV: +2.295 -> +4.045 -> +7.422.

This is sufficient evidence to continue the training line, not a final-strength claim.

HU is a tracked warning: LT1->LT2-A HU point estimates were flat/slightly worse across these weak families. Intervals remain wide and the HU policy reservoir is far below capacity, so do not declare a ceiling yet.

## Concurrent-fit optimization — CLOSED / ADMITTED

The repeated read-only fit screen compared sequential and concurrent 3H/HU Advantage fitting at 4 and 8 Torch threads.

Best admitted case:

- sequential 8-thread median combined fit wall: 7.1461 s;
- concurrent 8-thread median fit wall: 5.4460 s;
- fit speedup: 1.312188x;
- fit-wall reduction: about 23.79%.

Same-thread model hashes, final losses and batch RNG states matched exactly.

A full iteration-3001 concept gate passed exact semantic parity. The corrected production-function gate then passed with:

- `semantic_parity=true`;
- `source_unchanged=true`;
- `first_difference=null`;
- exact equality of models, optimizers, sampler/domain RNGs, counters, reservoir RNGs, added Advantage/Policy sample streams and non-timing report semantics.

See `LT2_PRODUCTION_CONCURRENT_PARITY_RESULT_20260917.md`.

This optimization branch is now closed. Do not repeat more fit-concurrency tuning before Stage B unless the workload materially changes.

## LT2 Stage B — READY

Canonical launcher: `tools/run_long_training_lt2_stage_b.sh`.

Continue from the preserved iteration-3000 checkpoint to iteration **7500**:

- +4500 iterations;
- +2.7M roots;
- 4.5M roots total;
- 31 root workers;
- 8 parent Torch threads;
- vectorized batches;
- production `concurrent_fit` iteration mode;
- checkpoint every 250 iterations;
- one-minute WSL memory/swap telemetry;
- isolated run directory; Stage A source remains read-only;
- final AveragePolicy fit and finalized checkpoint mandatory.

Why 7500: Stage A added about 276k HU AveragePolicy samples per 1000 iterations on average. From 820,667, approximately 1.18M more are needed to reach 2M, projecting the crossing near iteration 7.3k. Iteration 7500 gives a bounded margin to enter the full four-reservoir saturation/replacement state.

### Checkpoint cadence

Stage A used every 100 iterations and showed finalized saves around 54.5 s. Stage B widens this to 250 iterations to reduce serialization overhead while keeping restart loss bounded to at most one 250-iteration interval.

### Stage B completion review

After `LT2_STAGE_B_PASS`, stop. Review before any extension:

- final HU AveragePolicy seen/length and whether 2M was crossed;
- RAM/swap with all four 2M memories saturated or near saturated;
- checkpoint size/save-time trend;
- actual production throughput with concurrent fit;
- 3H weak-baseline learning trend;
- HU weak-baseline learning trend;
- any new serialization bottleneck;
- direct DeepCrusher benchmark if the faithful oracle is ready.

## Later LT2 / LT3

If Stage B remains healthy and strategic improvement continues, extend the same checkpoint through larger resumable blocks measured in millions and then tens/hundreds of millions of roots.

The exact final root count is not frozen. Continue while meaningful strategic improvement is still occurring and resources remain healthy. Stop or change architecture only when evidence indicates a plateau, regression, semantic defect or unacceptable compute efficiency.

## Strength tracking

Weak fixed opponents are regression sentinels, not the final product target.

Primary future strength evidence:

- SpinCore vs faithful DeepCrusher paired chip-EV;
- HU and 3H separately;
- blind/stack/position breakdowns;
- later full Spin & Go tournament win rate after continuous tournament progression is frozen.

DeepCrusher oracle construction proceeds in parallel. Do not use a simplified imitation for canonical head-to-head claims.

## Reservoir policy

Do not shrink the 2M reservoirs merely to save memory. Stage A proved the current contract healthy with zero swap use.

Do not increase beyond 2M without new evidence. Once all four memories are saturated, first evaluate learning quality and serialization/resource cost.

## Operational files

- `tools/run_long_training_lt1.sh` — historical LT1 launcher; do not use for active continuation.
- `tools/run_long_training_lt2_stage_a.sh` — completed Stage A launcher.
- `tools/benchmark_lean_lt2_concurrent_fit.sh` — closed optimization evidence.
- `tools/validate_lt2_production_concurrent_iteration.sh` — production parity PASS gate.
- `tools/run_long_training_lt2_stage_b.sh` — active bounded Stage B launcher.
- `tools/run_lean_functional_training.py` — authoritative trainer; `sequential` default, `concurrent_fit` admitted for Stage B.
- `tools/run_lean_learning_curve_eval.sh` — fixed-seed weak-baseline diagnostic.

## Immediate direction

1. Preserve iteration-3000 Stage A checkpoint.
2. Pull current `main`.
3. Run `bash tools/run_long_training_lt2_stage_b.sh`.
4. Let it finish to PASS or FAIL; do not manually interrupt for ordinary CPU-utilization oscillation.
5. On PASS, stop at iteration 7500 and review report + memory telemetry before any further training.
6. Continue faithful DeepCrusher oracle construction in parallel.
