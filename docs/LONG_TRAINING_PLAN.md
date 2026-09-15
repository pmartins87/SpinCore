# SpinCore — Long-Training Plan

Status: **CANONICAL TRAINING DIRECTION**
Date: 2026-09-15

## Purpose

The 120k-root run completed on 2026-09-15 is a functional calibration run, not an attempt to produce a final strong Spin & Go agent. It exists to prove that the repaired legacy-first pipeline can train, save, resume and play complete hands without the historical gross failures.

The target product is expected to require **orders of magnitude more learning** than this calibration. The project should be planned around sustained, Ryzen-optimized training over long periods, potentially weeks or months, with checkpoints and useful progress measurements along the way.

## Core rule

Do not judge the final strength ceiling of SpinCore from the 120k-root checkpoint. Results versus random/passive/jam baselines at this stage are smoke diagnostics only. They are useful for catching gross defects, not for deciding whether the learning architecture has reached its potential.

Likewise, the DeepCrusher benchmark is a **future strength/acceptance metric**, not a prerequisite for beginning serious training. Build that benchmark in parallel so it is ready when the policy has accumulated enough learning to make the comparison meaningful.

## Training phases

### Phase LT0 — calibration (DONE)

- 120k roots;
- mechanics and all-street play confirmed;
- Ryzen execution profile measured;
- weak-baseline diagnostics only.

This phase must not be confused with a competitive model.

### Phase LT1 — optimized scale validation

Before committing the machine for weeks/months, run a **short but production-shaped optimized block** large enough to measure real throughput, RAM growth, checkpoint size/time, reservoir behavior and restart safety under the selected Ryzen profile (31 root workers / 8 parent Torch threads).

The purpose is infrastructure validation, not strategy judgment.

### Phase LT2 — sustained training

Once LT1 confirms that long-run memory/checkpoint behavior is safe, move to continuous resumable training measured in **millions to tens/hundreds of millions of roots**, not thousands. The exact final total is not frozen in advance; training continues while strategically useful improvement is still occurring and resource use remains stable.

Long training should be checkpointed frequently enough to avoid large losses from interruption, but not so frequently that serialization becomes a major bottleneck.

### Phase LT3 — strength tracking

During sustained training, evaluate only at meaningful checkpoints. The main metrics are:

- direct SpinCore vs DeepCrusher paired chip-EV once the DeepCrusher oracle is faithful;
- HU and 3H separately;
- blind/stack/position breakdowns;
- later, full Spin & Go tournament win rate once continuous tournament progression is implemented;
- weak fixed opponents only as regression sentinels, not as the target.

The purpose is to observe learning progress and detect regressions/diminishing returns, not to interrupt training with constant certification exercises.

## Reservoir / memory requirement

The current 100k-sample reservoir per domain was acceptable for a calibration run. It must **not automatically be treated as the correct months-scale setting**. Before LT2, measure memory cost and retention behavior under the production observation/sample format and choose a capacity/representation that preserves useful diversity without wasting RAM.

If the present Python-object reservoir becomes the dominant memory/checkpoint bottleneck, replace it with a more compact storage representation rather than accepting a small reservoir merely for convenience.

## Benchmark timing

DeepCrusher benchmark construction continues now, because faithful OpenPPL parity takes engineering time. But the first extensive result should be interpreted as a **learning-curve checkpoint**, not as a verdict on a barely trained model.

The benchmark is most valuable after serious optimized training has begun. Early smoke runs may be used only to validate benchmark mechanics and identical-policy neutrality.

## Quality principle

DeepSpin previously spent roughly three months training and still produced gross mistakes. Therefore duration alone is not sufficient. The repaired SpinCore long run must combine:

- correct evaluator/state/action semantics;
- realistic 3H/HU/blind/stack sampling;
- repaired regret fallback;
- training/inference parity;
- Ryzen-optimized throughput;
- enough total learning volume.

The project goal is not "train for months because months sounds large". It is to make months of compute **actually useful** rather than repeat the old failure mode.

## Immediate direction

1. Treat the 120k model as LT0 calibration only.
2. Finish production-scale reservoir/checkpoint/throughput readiness.
3. Run LT1 optimized scale validation.
4. Start sustained LT2 training.
5. Continue building the DeepCrusher oracle in parallel, but do not make current weak-model benchmark outcomes a gate for starting serious training.
