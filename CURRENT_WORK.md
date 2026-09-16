# SpinCore Current Work

Date: 2026-09-16
Status: **LT1 1.2M ROOTS COMPLETE — CHECKPOINT PRESERVED — FIT CANDIDATE VALIDATED LOCALLY — RYZEN MEASUREMENT NEXT**

## Current authoritative update — 2026-09-16

Read [LT1 completion review](docs/LT1_COMPLETION_REVIEW_20260916.md) before new compute. LT1 completed 2,000 iterations / 1.2M roots with finalization and exit 0. Preserve its checkpoint at `/home/rz9/spincore_lean_functional/runs/long_training_lt1/20260915_181249/checkpoint.pt`. Advantage fitting accounts for 71.08% of reported wall time. Optimize that phase with a bounded resume/throughput comparison before a long extension; do not restart learning. Both policy reservoirs remain below capacity, so the full 8M-slot memory peak is unproven. Final-save telemetry and conflicting elapsed clocks remain explicit limitations. Earlier sections below retain LT0 history; the immediate milestone at the end is updated.

## Fit optimization prepared — 2026-09-16

See `docs/LT1_FIT_OPTIMIZATION_20260916.md`. Optional vectorized batches preserve LT1 semantics and reference remains default. Targeted tests and a local load/next-iteration smoke passed. Next physical action: `bash tools/benchmark_lean_lt1_fit.sh` in WSL. It reads the LT1 checkpoint without overwriting it, compares ten fixed fit cases at the real batch/step settings, verifies same-thread parity, executes one disposable resumed iteration, writes a report and stops within a 20-minute timeout. No LT2 auto-launch and no claimed Ryzen speedup yet.

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

## Ryzen optimization — root profile measured; LT1 neural-fit review open

Standing rule across all projects: any substantial workload assigned to the user's Ryzen must be optimized for that machine before long execution.

Measured 32-logical-thread profile:

- root workers: **31**;
- parent Torch threads: **8**;
- each worker Torch/OpenMP/BLAS threads: **1**.

Measured root-tree time fell from 33.38 s with one worker to 3.68 s with 31 workers. Parent fitting was fastest at 8 threads; 16 was worse.

The later `logical_cpus=1` line in the weak-baseline evaluation was only a display bug caused by calling `nproc` after setting `OMP_NUM_THREADS=1`. Local confirmation: `nproc=32`, CPUs online `0-31`. The evaluation still used 31 workers.

## LT0 behavioral smoke — PASS

5,000 offline self-play hands completed with no illegal action and all streets reached. Compared with the tiny 1,000-root pilot, the 120k policy was much less degenerate. This proves mechanics/learning activity, not final strength.

The later 1,000-scenario weak-opponent diagnostic showed statistically significant improvement over an untrained uniform-legal control in some overall/3H comparisons, while HU/passive-caller/jammer results remained noisy or weak. Correct interpretation: **the pipeline learned non-random poker structure, but the model is far too early to judge as a finished strategy.**

Do not use this diagnostic as a gate against starting serious training.

## Long-training path — MAIN PRIORITY

Canonical plan: `docs/LONG_TRAINING_PLAN.md`.

### LT1 — optimized scale validation

Before committing the Ryzen for weeks/months, run one production-shaped optimized block whose purpose is infrastructure measurement, not strategy judgment. Measure:

- real optimized throughput;
- RAM growth;
- reservoir behavior;
- checkpoint size and save time;
- restart/resume safety;
- whether root collection or fitting/checkpointing becomes the next bottleneck.

### Reservoir requirement before LT2

The 100k reservoir used in LT0 was acceptable for calibration, but it must not silently become the months-scale limit. LT0 already saw millions of samples, so before sustained training we must measure whether 100k retains enough diversity and whether the current Python-object storage wastes RAM/checkpoint bandwidth.

If needed, increase capacity or compact the reservoir representation before LT2. Do not sacrifice long-run learning quality merely to avoid this engineering step.

### LT2 — sustained serious training

Once LT1 is safe, begin continuous resumable training measured in **millions to tens/hundreds of millions of roots**, potentially weeks/months. The final total is not fixed in advance; continue while meaningful strategic improvement remains and resource behavior is healthy.

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
- Do not launch months-scale training before LT1 proves reservoir/checkpoint/memory stability.
- Do not accept a tiny 100k reservoir by default if it compromises long-run diversity.
- Do not run long low-utilization workloads on the Ryzen.
- Do not use an approximate DeepCrusher oracle for canonical claims.
- Do not discard the LT0 checkpoint; preserve it as the first learning-curve baseline.

## Immediate next milestone

Follow the finite next step in `docs/LT1_COMPLETION_REVIEW_20260916.md`: inspect and optimize neural fitting, validate resume and throughput on an isolated checkpoint copy, then extend the same LT1 learning state into LT2 with memory monitoring. No fresh LT1 rerun, repeated certification matrix, or early DeepCrusher verdict is required.
