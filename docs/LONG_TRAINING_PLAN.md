# SpinCore — Long-Training Plan

Status: **CANONICAL TRAINING DIRECTION**
Date: 2026-09-16

## Latest execution state

LT1 completed 1.2M roots; see `LT1_COMPLETION_REVIEW_20260916.md`. Fit optimization is implemented but awaits the bounded actual-Ryzen comparison described in `LT1_FIT_OPTIMIZATION_20260916.md`. Preserve LT1 and do not rerun the fresh launcher. The active next action is the read-only-checkpoint fit benchmark; afterward extend into a separate LT2 directory.

## Purpose

The 120k-root run completed on 2026-09-15 is a functional calibration run, not an attempt to produce a final strong Spin & Go agent. It exists to prove that the repaired legacy-first pipeline can train, save, resume and play complete hands without the historical gross failures.

The target product is expected to require **orders of magnitude more learning** than this calibration. The project should be planned around sustained, Ryzen-optimized training over long periods, potentially weeks or months, with checkpoints and useful progress measurements along the way.

## Core rule

Do not judge the final strength ceiling of SpinCore from the 120k-root checkpoint. Results versus random/passive/jam baselines at this stage are smoke diagnostics only. They are useful for catching gross defects, not for deciding whether the learning architecture has reached its potential.

Likewise, the DeepCrusher benchmark is a **future strength/acceptance metric**, not a prerequisite for beginning serious training. Build that benchmark in parallel so it is ready when the policy has accumulated enough learning to make the comparison meaningful.

## Hardware contract

The production Ryzen has **64 GiB of physical RAM**, 32 logical CPUs, and a measured SpinCore execution profile of **31 root workers + 8 parent Torch threads**. Long training is therefore allowed to use materially larger replay reservoirs than the 100k calibration setting.

WSL may expose less memory than the physical DIMMs. Production launchers must size from `/proc/meminfo`, not from the Windows hardware specification. The LT1 launcher refuses to start if WSL exposes less than 28 GiB or if the training filesystem has less than 30 GiB free. We do not silently shrink the reservoir merely to make a run start, because that would trade away training diversity for convenience.

## Training phases

### Phase LT0 — calibration (DONE)

- 120k roots;
- mechanics and all-street play confirmed;
- Ryzen execution profile measured;
- weak-baseline diagnostics only.

This phase must not be confused with a competitive model.

### Phase LT1 — production-shaped scale milestone

LT1 is not a strength benchmark. It is the **beginning of the long-training campaign under production-like memory/optimizer pressure**, and if healthy its checkpoint is preserved and extended into LT2 rather than discarded.

Current LT1 profile:

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
- checkpoint serialization time and file size are measured on every save;
- `/usr/bin/time -v` records peak RSS, CPU and swap behavior.

The 2M setting intentionally returns to the scale used by mature DeepSpin reservoirs instead of freezing the tiny 100k calibration limit. With 64 GiB physical RAM this is a reasonable first production capacity, but LT1 still measures actual WSL RSS/checkpoint behavior before we commit to larger capacities.

### Phase LT2 — sustained training

Once LT1 confirms that long-run memory/checkpoint behavior is safe, continue the **same large-reservoir campaign** through resumable extensions rather than restarting. Training is measured in millions, then tens/hundreds of millions of roots.

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

The current Python reservoir stores compact SPNNIV1 observations but still pays Python-object overhead. LT1 therefore measures the actual checkpoint/RSS cost at production scale.

If 2M-per-memory-per-domain is comfortably below the WSL/RAM budget and checkpoint cost is acceptable, LT2 may preserve it or increase it. If Python-object overhead or serialization becomes the dominant bottleneck, the correct response is to implement a more compact packed reservoir/checkpoint format, **not** to fall back to a tiny 100k reservoir simply because it is easier.

The existing 120k checkpoint cannot recover samples that its 100k reservoirs already discarded, so the serious large-reservoir campaign starts fresh. LT0 remains preserved as a historical calibration baseline; LT1/LT2 is the production learning line.

## Optimizer-scale requirement

A months-scale campaign cannot simply reuse the tiny calibration fit budget without scrutiny. LT1 therefore raises the advantage fit to 100 × 1,024 examples per domain per iteration and trains the AveragePolicy for 4,000 × 1,024 examples per domain at the milestone. The purpose is to avoid generating millions of high-quality traversal samples while undertraining the neural approximators that must absorb them.

Advantage fitting still follows the current repaired Deep-CFR path; LT1 does not silently alter poker semantics, game sampling, action abstraction or utility. Any future change to advantage reset/warm-start semantics must be justified separately rather than mixed into a hardware optimization.

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

- `tools/run_long_training_lt1.sh` — start the fresh production-shaped 1.2M-root campaign after RAM/disk preflight.
- `tools/resume_long_training_lt1.sh` — resume/extend the latest LT1 checkpoint without restarting.
- `tools/run_lean_functional_training.py` — authoritative trainer; now reports checkpoint serialization time and size.
- `runs/long_training_lt1/<timestamp>/` — checkpoint/report/log location on the Ryzen.

## Immediate direction

1. Preserve LT0 unchanged.
2. Confirm how much of the 64 GiB is actually visible inside WSL.
3. Start LT1 with the 2M reservoirs and production 31/8 Ryzen profile.
4. Inspect real throughput, peak RSS, swap, checkpoint size/time and neural-fit cost.
5. If healthy, extend the same checkpoint into LT2 instead of restarting.
6. Continue building the DeepCrusher oracle in parallel; do not use the current barely trained policy as a product-strength verdict.
