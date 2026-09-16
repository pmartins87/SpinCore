# SpinCore — Long-Training Plan

Status: **CANONICAL TRAINING DIRECTION**
Date: 2026-09-16

## Latest execution state

LT1 completed 1.2M roots; see `LT1_COMPLETION_REVIEW_20260916.md`. The physical Ryzen neural-fit benchmark also passed; see `LT1_FIT_BENCHMARK_RESULT_20260916.md`. The selected execution profile is now 31 root workers, 8 parent Torch threads and vectorized batch construction. Preserve the finalized LT1 checkpoint and do not rerun the fresh launcher or the closed fit matrix.

The active next action is **LT2 Stage A**, a bounded continuation of the same learning state for exactly 1,000 additional iterations / 600,000 roots in a separate run directory. Use `tools/run_long_training_lt2_stage_a.sh`; after it stops, inspect memory/swap, reservoir saturation, checkpoint serialization and throughput before authorizing a larger LT2 block.

## Purpose

The 120k-root run completed on 2026-09-15 is a functional calibration run, not an attempt to produce a final strong Spin & Go agent. It exists to prove that the repaired legacy-first pipeline can train, save, resume and play complete hands without the historical gross failures.

The target product is expected to require **orders of magnitude more learning** than this calibration. The project should be planned around sustained, Ryzen-optimized training over long periods, potentially weeks or months, with checkpoints and useful progress measurements along the way.

## Core rule

Do not judge the final strength ceiling of SpinCore from the 120k-root checkpoint or the 1.2M-root LT1 infrastructure milestone. Results versus random/passive/jam baselines at this stage are smoke diagnostics only. They are useful for catching gross defects, not for deciding whether the learning architecture has reached its potential.

Likewise, the DeepCrusher benchmark is a **future strength/acceptance metric**, not a prerequisite for beginning serious training. Build that benchmark in parallel so it is ready when the policy has accumulated enough learning to make the comparison meaningful.

## Hardware contract

The production Ryzen has **64 GiB of physical RAM**, 32 logical CPUs, and a measured SpinCore execution profile of **31 root workers + 8 parent Torch threads**. The LT1 physical benchmark additionally selected **vectorized batch construction** at the real 100-step × 1,024-batch fit workload. Long training is therefore allowed to use materially larger replay reservoirs than the 100k calibration setting.

WSL may expose less memory than the physical DIMMs. Production launchers must size from `/proc/meminfo`, not from the Windows hardware specification. LT2 Stage A refuses to start if WSL exposes less than 28 GiB or if the training filesystem has less than 15 GiB free. We do not silently shrink the reservoir merely to make a run start, because that would trade away training diversity for convenience.

## Training phases

### Phase LT0 — calibration (DONE)

- 120k roots;
- mechanics and all-street play confirmed;
- Ryzen execution profile measured;
- weak-baseline diagnostics only.

This phase must not be confused with a competitive model.

### Phase LT1 — production-shaped scale milestone (DONE)

LT1 is not a strength benchmark. It is the **beginning of the long-training campaign under production-like memory/optimizer pressure**, and its checkpoint is preserved and extended into LT2 rather than discarded.

Completed LT1 profile:

- 2,000 iterations;
- 600 roots/iteration = **1.2 million roots**;
- same empirical 3H/HU/blind/stack sampler as LT0;
- 31 independent root workers;
- 8 parent Torch threads;
- reservoir capacity **2,000,000 samples for each memory in each domain**;
- maximum reservoir slots: 8,000,000 total (2 domains × advantage/policy × 2M);
- 100 advantage optimizer steps per domain per iteration;
- batch size 1,024;
- 4,000 AveragePolicy optimizer steps per domain at the milestone finalization;
- checkpoint every 100 iterations;
- checkpoint serialization time and file size measured on every save;
- `/usr/bin/time -v` used for process-level resource telemetry.

LT1 completed 1.2M roots successfully. Both advantage memories reached capacity. Policy memories ended at 1,485,285 samples for 3H and 544,490 for HU, so LT1 did not yet establish the fully occupied four-memory state. Advantage fitting consumed 71.08% of reported measured wall time, motivating the bounded neural-fit optimization gate.

### LT1 neural-fit optimization (DONE)

The physical benchmark compared reference/vectorized batching under 1/2/4/8/16 parent Torch threads using the real finalized LT1 reservoirs, unchanged 100 fit steps per domain and batch size 1,024.

Baseline 8-thread/reference combined fit time was `7.663904482004 s`. The fastest eligible case was 8-thread/vectorized at `6.5824409390043 s`, a `1.1642952140430884x` fit speedup and 14.11% fit-time reduction. Same-thread final losses and model hashes matched. A disposable resumed iteration 2001 completed with 31 root workers and 600 roots total, and the source LT1 checkpoint remained byte-identical.

Freeze for the next stage:

```text
root workers = 31
worker Torch/OpenMP/BLAS threads = 1
parent Torch threads = 8
batch_mode = vectorized
```

Do not repeat the matrix absent new evidence that materially changes the workload.

### Phase LT2 — sustained training

LT2 continues the **same large-reservoir campaign** through resumable extensions rather than restarting. Training is ultimately expected to be measured in millions, then tens/hundreds of millions of roots, but the first continuation is deliberately bounded.

#### LT2 Stage A — first policy-reservoir saturation gate

Fixed contract:

- start from the preserved finalized LT1 checkpoint at iteration 2000;
- copy it into a separate `runs/long_training_lt2/<timestamp>/` directory before training;
- verify source SHA256 `beef9bee9439de8d9153450d190e62b0388a678b0778c4185108361f626b4337` before and after;
- run iterations **2001–3000** only;
- 1,000 additional iterations = **600,000 additional roots**;
- 31 root workers;
- 8 parent Torch threads;
- vectorized batches;
- checkpoint every 100 iterations;
- one-minute WSL memory/swap telemetry;
- stop after completion for review; no automatic larger continuation.

Why 1,000 iterations: at LT1's observed strategy-sample rate, 3H should add roughly 0.74M policy samples and therefore reach its 2M reservoir capacity during Stage A, while HU should remain below capacity. This makes Stage A large enough to exercise the first policy-memory saturation transition but small enough to stop before committing to a much longer run. The exact crossing point is only a planning estimate.

Stage A review must inspect:

- successful continuation through iteration 3000 with finite losses;
- source LT1 checkpoint unchanged;
- WSL `MemAvailable` and swap behavior across the run;
- 3H/HU reservoir counts and which memories reached capacity;
- checkpoint size and save time as 3H policy memory fills;
- actual vectorized fit time, root-tree time and policy-sampling time;
- any new serialization, memory, or CPU bottleneck.

If healthy, continue the **same LT2 checkpoint** in larger resumable blocks. If Python-object reservoir overhead or checkpoint serialization becomes dominant, implement a more compact packed representation/checkpoint format rather than shrinking the 2M capacity merely for convenience.

The exact final root count is not frozen. Training continues while strategically useful improvement is still occurring and resource use remains stable. If the optimized implementation reaches an equivalent amount of learning in days rather than months, we do not artificially waste calendar time; the target is useful learning volume and poker strength, not a ritual number of days.

### Phase LT3 — strength tracking

During sustained training, evaluate only at meaningful checkpoints. The main metrics are:

- direct SpinCore vs DeepCrusher paired chip-EV once the DeepCrusher oracle is faithful;
- HU and 3H separately;
- blind/stack/position breakdowns;
- later, full Spin & Go tournament win rate once continuous tournament progression is implemented;
- weak fixed opponents only as regression sentinels, not as the target.

The purpose is to observe the learning curve and detect regressions/diminishing returns, not to interrupt training with constant certification exercises.

## Reservoir / memory requirement

The current Python reservoir stores compact SPNNIV1 observations but still pays Python-object overhead. LT1 showed that the advantage memories can reach the 2M limit, but both policy memories were not yet full. LT2 Stage A deliberately exercises the 3H policy transition before larger blocks.

If 2M-per-memory-per-domain remains comfortably below the WSL/RAM budget and checkpoint cost is acceptable, LT2 should preserve it. Any increase beyond 2M requires new evidence; it is not automatically authorized by the current benchmark. If Python-object overhead or serialization becomes dominant, the correct response is to implement a more compact packed reservoir/checkpoint format, **not** to fall back to a tiny 100k reservoir simply because it is easier.

The existing 120k checkpoint cannot recover samples that its 100k reservoirs already discarded, so the serious large-reservoir campaign is the LT1/LT2 line. LT0 remains preserved as a historical calibration baseline.

## Optimizer-scale requirement

A months-scale campaign cannot simply reuse the tiny calibration fit budget without scrutiny. LT1 raised the advantage fit to 100 × 1,024 examples per domain per iteration and trains the AveragePolicy for 4,000 × 1,024 examples per domain at milestone finalization. The physical benchmark optimized only batch construction and thread count; it did **not** reduce steps, batch size, memory capacity, change optimizer/loss, alter reset/RNG semantics or change poker sampling.

Advantage fitting still follows the current repaired Deep-CFR path. Future changes to advantage reset/warm-start semantics, optimizer scale or concurrent-domain fitting require separate evidence rather than being mixed into hardware optimization.

## Benchmark timing

DeepCrusher benchmark construction continues now, because faithful OpenPPL parity takes engineering time. But early head-to-head smoke runs are only for benchmark mechanics. The extensive DeepCrusher comparison becomes strategically meaningful after sustained training has accumulated enough learning.

## Quality principle

DeepSpin previously spent roughly three months training and still produced gross mistakes. Therefore duration alone is not sufficient. The repaired SpinCore long run must combine:

- correct evaluator/state/action semantics;
- realistic 3H/HU/blind/stack sampling;
- repaired regret fallback;
- training/inference parity;
- Ryzen-optimized throughput;
- large enough reservoirs;
- enough neural fitting to absorb reservoir information;
- enough total learning volume.

The project goal is not "train for months because months sounds large". It is to make long-running compute **actually useful** rather than repeat the old failure mode.

## Operational files

- `tools/run_long_training_lt1.sh` — historical fresh LT1 production-shaped launcher; do not rerun for the active line.
- `tools/resume_long_training_lt1.sh` — historical same-directory LT1 resume helper; do not use it for LT2 because LT2 must preserve the original LT1 checkpoint separately.
- `tools/benchmark_lean_lt1_fit.sh` — closed physical fit benchmark; rerun only if a materially changed workload requires a new gate.
- `tools/run_long_training_lt2_stage_a.sh` — canonical next launcher; isolated LT1 copy, iterations 2001–3000, memory telemetry, stop-after-stage behavior.
- `tools/run_lean_functional_training.py` — authoritative trainer; records batch mode, Torch threads, checkpoint serialization and final measured wall scope.
- `runs/long_training_lt1/20260915_181249/` — preserved LT1 source run.
- `runs/long_training_lt2/<timestamp>/` — LT2 continuation runs.

## Immediate direction

1. Preserve LT0 and the finalized LT1 checkpoint unchanged.
2. Pull current `main` on the Ryzen.
3. Run `bash tools/run_long_training_lt2_stage_a.sh` once.
4. Wait for `LT2_STAGE_A_PASS`; do not start another block automatically.
5. Review `report.json`, `memory.log`, checkpoint metrics and reservoir state.
6. If Stage A is healthy, authorize a larger continuation from the LT2 checkpoint; otherwise address the measured bottleneck first.
7. Continue building the DeepCrusher oracle in parallel; do not use the current early policy as a final product-strength verdict.
