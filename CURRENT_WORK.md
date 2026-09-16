# SpinCore Current Work

Date: 2026-09-16
Status: **LT1 1.2M ROOTS COMPLETE — RYZEN FIT GATE PASS — LT2 STAGE A READY**

## Current authoritative update — 2026-09-16

Read [LT1 completion review](docs/LT1_COMPLETION_REVIEW_20260916.md) and [LT1 physical fit benchmark result](docs/LT1_FIT_BENCHMARK_RESULT_20260916.md) before new compute. LT1 completed 2,000 iterations / 1.2M roots with finalization and exit 0. Preserve its checkpoint at `/home/rz9/spincore_lean_functional/runs/long_training_lt1/20260915_181249/checkpoint.pt`, SHA256 `beef9bee9439de8d9153450d190e62b0388a678b0778c4185108361f626b4337`. The bounded physical Ryzen benchmark passed: 8 parent Torch threads + vectorized batching reduced combined 3H/HU fit time from 7.66390 s to 6.58244 s, a 1.1643x fit speedup, with same-thread loss/model parity and a successful disposable 31-worker resume iteration. Do not repeat the matrix. The next finite compute gate is LT2 Stage A: continue the preserved LT1 state for exactly 1,000 more iterations / 600,000 roots in a separate run directory using `tools/run_long_training_lt2_stage_a.sh`, then stop for memory/reservoir/checkpoint review before any larger block.

## Fit optimization — PHYSICAL PASS 2026-09-16

See `docs/LT1_FIT_OPTIMIZATION_20260916.md` for the implementation contract and `docs/LT1_FIT_BENCHMARK_RESULT_20260916.md` for the physical result. The winner is frozen for the next stage at 31 root workers, 8 parent Torch threads and `batch_mode=vectorized`. The benchmark source checkpoint remained unchanged, iteration 2001 completed with exactly 600 roots, and no long training was started by the benchmark. LT2 Stage A exists specifically to cross the first policy-reservoir saturation transition under the optimized execution path before authorizing tens of millions of roots.

## Goal

Make the multi-year DeepSpin project actually work as SpinCore. Preserve mature legacy knowledge; replace only components with a concrete correctness, learning-quality, or compute-efficiency reason.

A product-quality acceptance metric remains explicit: **SpinCore must eventually beat DeepCrusher in extensive fair offline simulation under common game semantics.** But that benchmark is a future strength/acceptance metric, not a prerequisite for starting serious training and not a sensible verdict on a barely trained model.

The long-training direction is canonical in `docs/LONG_TRAINING_PLAN.md`. Poker-level explanation is in `docs/POKER_GOALS_AND_TRAINING_EXPLAINED.md`. The direct head-to-head contract is in `docs/DEEPC_RUSHER_BENCHMARK_SPEC.md`.

## Important correction — 120k was calibration, not serious strength training

The completed 120,000-root run must be understood as **LT0 calibration only**. It was designed to prove that the repaired pipeline can train, save, resume and play complete hands across the real 3H/HU/blind distribution without the historical gross failures.

It was **not** intended to create a DeepCrusher-beating final policy. Results versus `UNIFORM_LEGAL`, `PASSIVE_CALLER` and `JAMMER` are therefore smoke diagnostics only. They can reveal catastrophic defects, but they must not be used as evidence that the architecture has already reached its strategic ceiling.

The project's expected serious learning scale is orders of magnitude larger: sustained optimized training in millions to tens/hundreds of millions of roots, potentially over weeks or months, while useful improvement continues.

Do not repeat the mistake of treating a short calibration policy as if it should already be strong.

## First functional SpinCore — decisions closed

- real SpinGo scenario distribution: 3-handed + true HU, blind ladder 10/20 through 100/200, separate empirical blind weights, blind-conditioned stack distributions, 1500 total chips;
- one WTA/chip-EV policy family first;
- terminal utility `chip_delta / 1500`;
- compact SPNNIV1 exact-state-derived representation;
- mature DeepSpin seven-action vocabulary with contextual preflop semantics and 33/50/75/100% postflop sizes plus all-in;
- external-sampling Deep CFR (`exact_opponent_levels=0`);
- sampled AveragePolicy trajectories;
- repaired all-nonpositive regret fallback;
- separate 3H and HU brains.

## LT0 calibration — COMPLETED

Training contract:

- 200 iterations;
- 600 roots/iteration = **120,000 roots**;
- 65,400 3H + 54,600 HU roots;
- 20,838,215 traversal nodes;
- 3,806,986 advantage samples seen;
- 213,935 AveragePolicy samples seen;
- final AveragePolicy fit completed;
- empirical blind ladder genuinely exercised;
- checkpoint preserved at `/home/rz9/spincore_lean_functional/runs/lean_first_training/20260915_131133/checkpoint.pt`.

The serial launch took about **1h25m** because it used only about two logical CPUs. That was a process mistake; it is not the production execution profile.

## Ryzen optimization — physical fit profile closed

Standing rule across all projects: any substantial workload assigned to the user's Ryzen must be optimized for that machine before long execution.

Frozen next-stage profile:

- root workers: **31**;
- parent Torch threads: **8**;
- each worker Torch/OpenMP/BLAS threads: **1**;
- neural batch construction: **vectorized**.

Measured root-tree time previously fell from 33.38 s with one worker to 3.68 s with 31 workers. The LT1 physical fit benchmark then confirmed 8 threads as the best tested parent count at the real 100-step × 1024-batch workload: vectorized 8-thread fit took 6.58244 s combined across 3H/HU, versus 8.47159 s at 16 threads and 14.88826 s at one thread. The same-thread reference/vectorized winner preserved final losses and model hashes.

The later `logical_cpus=1` line in the weak-baseline evaluation was only a display bug caused by calling `nproc` after setting `OMP_NUM_THREADS=1`. Local confirmation: `nproc=32`, CPUs online `0-31`. The evaluation still used 31 workers.

## LT0 behavioral smoke — PASS

5,000 offline self-play hands completed with no illegal action and all streets reached. Compared with the tiny 1,000-root pilot, the 120k policy was much less degenerate. This proves mechanics/learning activity, not final strength.

The later 1,000-scenario weak-opponent diagnostic showed statistically significant improvement over an untrained uniform-legal control in some overall/3H comparisons, while HU/passive-caller/jammer results remained noisy or weak. Correct interpretation: **the pipeline learned non-random poker structure, but the model is far too early to judge as a finished strategy.**

Do not use this diagnostic as a gate against starting serious training.

## Long-training path — MAIN PRIORITY

Canonical plan: `docs/LONG_TRAINING_PLAN.md`.

### LT1 — optimized scale validation — COMPLETE

LT1 completed 2,000 iterations / 1.2M roots. It established the production-shaped large-reservoir line, measured checkpoint cost and exposed neural fitting as the dominant measured phase. The subsequent physical benchmark closed the immediate fit bottleneck screen without changing learning semantics.

### Reservoir requirement entering LT2

The LT1 reservoir capacity is 2M per memory per domain. Both advantage memories reached capacity. Policy memories ended at 1,485,285 3H and 544,490 HU samples, so LT1 did not yet measure the fully occupied four-memory state. LT2 Stage A is deliberately sized so the 3H policy memory should reach capacity during the block, allowing memory/checkpoint behavior to be reviewed at the first saturation transition.

Do not shrink the 2M reservoirs merely to simplify runtime. If serialization or Python-object overhead becomes the bottleneck, prefer a compact representation/checkpoint engineering step.

### LT2 — sustained serious training

LT2 begins by extending the **same LT1 learning state**, not by restarting. Stage A is fixed at +1,000 iterations / +600,000 roots and must stop for review. If memory, swap, checkpoint serialization and throughput remain healthy, continue the same LT2 checkpoint through larger resumable blocks measured in millions and eventually tens/hundreds of millions of roots while meaningful strategic improvement continues.

Checkpoints must be frequent enough to avoid large losses from interruption, but sparse enough not to become a throughput bottleneck.

## DeepCrusher benchmark — BUILD IN PARALLEL, DO NOT USE CURRENT POLICY AS FINAL VERDICT

Frozen first opponent: DeepCrusher R8 v22 good/stable.

Already implemented:

- frozen source/hash contract;
- exact external Fold/Check/Call/BetTo/RaiseTo/AllIn bridge without quantizing DeepCrusher into SpinCore sizes;
- HU paired same-deal seat swap;
- balanced 3H AAB/ABB blocks with seat rotation;
- zero-sum accounting;
- observable-state bridge;
- OpenPPL dependency inventory;
- identical-policy neutrality infrastructure.

Remaining central task is the faithful DeepCrusher decision oracle. It must reproduce the frozen OpenPPL strategy, not a simplified caricature.

Build this now because the engineering takes time. But extensive SpinCore-vs-DeepCrusher results become strategically meaningful only after serious LT2 training has accumulated. Early head-to-head runs are benchmark-mechanics smokes, not product verdicts.

Future benchmark stages:

1. DC0 oracle/source/runtime parity;
2. DC1 small balanced mechanics smoke;
3. DC2 extensive paired chip-EV after serious training checkpoints;
4. DC3 full Spin & Go tournament win rate after continuous tournament progression is frozen.

## Historical lesson

Old DeepSpin trained for roughly three months and still made gross errors. Therefore "months" alone is not enough. The point of the current architecture work is to ensure that months of Ryzen compute are **useful**: correct state/evaluator semantics, realistic sampling, repaired regret behavior, inference parity, appropriate memory and high hardware throughput.

## Do not do

- Do not call the 120k policy final or expect it to beat DeepCrusher already.
- Do not let the weak-baseline diagnostic block serious training.
- Do not restart LT1 or rerun the closed fit matrix.
- Do not jump directly to tens of millions of roots before LT2 Stage A reviews the first policy-reservoir saturation transition.
- Do not accept a tiny 100k reservoir by default if it compromises long-run diversity.
- Do not run long low-utilization workloads on the Ryzen.
- Do not use an approximate DeepCrusher oracle for canonical claims.
- Do not discard LT0 or the preserved LT1 checkpoint.

## Immediate next milestone

Run `tools/run_long_training_lt2_stage_a.sh` on the Ryzen after pulling `main`. It verifies the exact preserved LT1 checkpoint hash, copies it into a separate `runs/long_training_lt2/<timestamp>/` directory, continues iterations 2001–3000 with 31 workers / 8 Torch threads / vectorized batches, records one-minute WSL memory/swap telemetry, saves every 100 iterations and stops. After `LT2_STAGE_A_PASS`, review `report.json`, `memory.log` and checkpoint metrics before authorizing any larger LT2 block.
