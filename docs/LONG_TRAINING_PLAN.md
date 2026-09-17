# SpinCore — Long-Training Plan

Status: **CANONICAL TRAINING DIRECTION**
Date: 2026-09-16

## Current state

The active learning line is continuous:

- LT0: 120k roots — calibration only;
- LT1: 1.2M roots — production-shaped scale milestone;
- LT2 Stage A: 1.8M roots total — resource/saturation gate PASS;
- next training milestone after the current fit screen: approximately 4.5M roots total at iteration 7500.

Read `LT2_STAGE_A_REVIEW_20260916.md` for the current evidence.

## Core training contract

The functional line uses:

- the empirical SpinGo 3H/HU/blind/stack sampler;
- WTA chip-EV utility scaled by 1500;
- SPNNIV1 frozen-control representation for the current functional line;
- the mature legacy action vocabulary;
- external-sampling Deep CFR with repaired all-nonpositive regret fallback;
- separate 3H and HU brains;
- sampled AveragePolicy trajectories;
- 2,000,000-sample reservoir capacity per memory per domain;
- 600 roots per iteration;
- 100 Advantage optimizer steps per domain per iteration;
- batch size 1024;
- 4000 AveragePolicy optimizer steps per domain at milestone finalization.

The current proven Ryzen execution profile is:

- 31 root workers;
- one numerical-library thread per worker;
- 8 parent Torch threads;
- vectorized neural batch construction.

## LT0 — calibration — DONE

120k roots proved that the repaired pipeline can train, save, resume and play complete hands across the real domain distribution. It is not a competitive-strength checkpoint.

Preserve:

`/home/rz9/spincore_lean_functional/runs/lean_first_training/20260915_131133/checkpoint.pt`

## LT1 — production-shaped milestone — DONE

LT1 completed 2000 iterations / 1.2M roots and finalization successfully.

Preserve:

`/home/rz9/spincore_lean_functional/runs/long_training_lt1/20260915_181249/checkpoint.pt`

SHA256:

`beef9bee9439de8d9153450d190e62b0388a678b0778c4185108361f626b4337`

LT1 established the 2M reservoir line and exposed neural Advantage fitting as the dominant measured phase. The physical fit benchmark then selected 8 parent Torch threads plus vectorized batch construction, with a 1.1643x fit-throughput gain over the 8-thread/reference baseline and exact same-thread model/loss parity.

Do not rerun that matrix absent a materially changed workload.

## LT2 Stage A — first policy-reservoir saturation gate — PASS

Stage A continued the exact LT1 state through iterations 2001–3000, adding 600k roots and reaching 1.8M total.

Current checkpoint:

`/home/rz9/spincore_lean_functional/runs/long_training_lt2/20260916_015559/checkpoint.pt`

SHA256:

`e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`

Resource result:

- wall scope about 2h55m47s;
- min WSL MemAvailable about 10.67 GiB;
- swap used 0;
- process max RSS about 16.3 GiB;
- final checkpoint 2.236 GiB;
- final save about 54.5 s;
- exit 0 / `LT2_STAGE_A_PASS`.

Reservoir result:

- both Advantage reservoirs are saturated;
- 3H AveragePolicy crossed the 2M cap and is in replacement regime;
- HU AveragePolicy is 820,667 and remains below capacity.

Checkpoint growth slowed materially once the 3H policy reservoir saturated, which is the expected healthy behavior rather than runaway serialization growth.

## Learning evidence after Stage A

The same 1000 fixed-seed scenarios were used to compare 120k, 1.2M and 1.8M checkpoints against uniform-legal, passive-caller and jammer families.

The useful directional trend is positive overall and especially in 3H:

- uniform overall cEV: +10.836 -> +15.873 -> +17.015;
- uniform 3H cEV: +2.407 -> +8.898 -> +12.769;
- passive-caller overall cEV: -2.402 -> -0.553 -> -0.064;
- jammer overall cEV: -5.498 -> -2.719 -> -1.298;
- jammer 3H cEV: +2.295 -> +4.045 -> +7.422.

This is enough evidence to continue the training line. It is not a final-strength claim.

HU is a tracked warning. The LT1->LT2-A HU point estimates were flat/slightly worse across these weak baselines. The current diagnostic has wide HU intervals and does not provide a dedicated paired CI for checkpoint-vs-checkpoint deltas. Because the HU policy reservoir is still far below 2M, do not declare a ceiling yet.

## Immediate compute gate — concurrent-fit screen

Stage A confirmed that average CPU utilization remains low because the two domain Advantage fits are serial and fitting remains the dominant phase.

Before a roughly half-day Stage B run, execute one bounded read-only screen:

`tools/benchmark_lean_lt2_concurrent_fit.sh`

It compares sequential and concurrent 3H/HU Advantage fitting at 4 and 8 Torch threads. Resets remain sequential to avoid racing the forked Torch RNG initialization scope.

A concurrent candidate is worth integrating only if:

- same-thread sequential/concurrent model hashes match exactly for both domains;
- final losses match exactly;
- batch RNG states match exactly;
- the source checkpoint remains unchanged;
- median combined fit wall improves by at least 5% versus sequential 8-thread fitting.

If the screen does not clear 5%, stop tuning and keep the current 31/8/vectorized execution path. If it clears 5%, build one full-iteration candidate and prove exact learning-state parity before production use.

## LT2 Stage B — next bounded milestone

After the fit execution path is cleared, continue from the preserved iteration-3000 checkpoint to approximately iteration 7500:

- +4500 iterations;
- +2.7M roots;
- 4.5M roots total.

Why approximately 7500: Stage A added about 276k HU AveragePolicy samples over 1000 iterations. From 820,667, roughly 1.18M more are needed to reach 2M, implying about 4.3k more iterations at the observed rate. Iteration ~7500 therefore gives margin to cross the HU policy-reservoir saturation transition.

Stage B must remain resumable and bounded. Do not jump directly to tens/hundreds of millions of roots before reviewing the all-reservoir-saturated state.

At Stage B completion review:

- RAM/swap with all four 2M memories effectively saturated;
- checkpoint size and save time;
- actual throughput;
- 3H weak-baseline learning trend;
- HU weak-baseline learning trend;
- any new serialization bottleneck;
- DeepCrusher direct benchmark if the faithful oracle is ready.

## Later LT2 / LT3

If Stage B remains healthy and strategic improvement continues, extend the same checkpoint through larger resumable blocks measured in millions and then tens/hundreds of millions of roots.

The exact final root count is not frozen. Continue while meaningful strategic improvement is still occurring and resource use is healthy. Stop or change architecture only when evidence indicates a plateau, regression, semantic defect or unacceptable compute efficiency.

## Strength tracking

Weak fixed opponents are regression sentinels. They are not the final product target.

Primary future strength evidence:

- SpinCore vs faithful DeepCrusher paired chip-EV;
- HU and 3H separately;
- blind/stack/position breakdowns;
- later full Spin & Go tournament win rate after continuous tournament progression is frozen.

DeepCrusher oracle construction proceeds in parallel. Do not use a simplified imitation for canonical head-to-head claims.

## Reservoir policy

Do not shrink the 2M reservoirs merely to save memory. Stage A proved the current contract is healthy under WSL memory pressure and no swap was used.

Do not increase beyond 2M without new evidence. Once all four memories are saturated, the first question is whether learning quality and checkpoint/serialization cost remain acceptable, not whether larger reservoirs sound better.

## Checkpoint policy

Checkpoint frequency should balance restart loss against serialization overhead. Stage A's 100-iteration cadence was intentionally conservative for the first saturation gate. Longer Stage B blocks may use a wider cadence after explicit launcher review, provided final checkpoints and milestone preservation remain mandatory.

## Operational files

- `tools/run_long_training_lt1.sh` — historical fresh LT1 launcher; do not use for active continuation.
- `tools/run_long_training_lt2_stage_a.sh` — completed Stage A launcher.
- `tools/benchmark_lean_lt2_concurrent_fit.sh` — immediate bounded fit-concurrency screen.
- `tools/run_lean_functional_training.py` — authoritative functional trainer.
- `tools/run_lean_learning_curve_eval.sh` — fixed-seed weak-baseline learning-curve diagnostic.

## Immediate direction

1. Preserve the iteration-3000 LT2 Stage A checkpoint.
2. Run the concurrent-fit benchmark exactly once.
3. Do not launch Stage B automatically.
4. Review the benchmark report.
5. If concurrency is not worthwhile, retain the current execution profile and prepare Stage B.
6. If concurrency is worthwhile, validate one full-iteration exact-parity implementation before Stage B.
7. Continue DeepCrusher oracle construction in parallel.
