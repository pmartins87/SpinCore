# SpinCore — Long-Training Plan

Status: **LT2 STAGE B PASS — TRAINING PAUSED AT 4.5M ROOTS — VARIANCE-FIRST WEAK-BASELINE GATE ACTIVE**
Date: 2026-09-17

## Current state

The continuous learning line has reached:

- LT0: 120k roots — calibration;
- LT1: 1.2M roots — production-shaped milestone;
- LT2 Stage A: 1.8M roots;
- LT2 Stage B: 4.5M roots / iteration 7500;
- all four 2M reservoirs in replacement regime;
- Stage A -> Stage B policy movement is material;
- existing weak-baseline and checkpoint-strength evaluations are too noisy to justify another long block or an architecture change;
- immediate next gate is a 30k multi-seed weak-baseline evaluation with explicit uncertainty control.

Read `LT2_VARIANCE_AND_WEAK_BASELINE_GATE_20260917.md` first.

## Core training contract

Current functional line:

- empirical SpinGo 3H/HU/blind/stack sampling;
- WTA chip-EV utility scaled by 1500;
- SPNNIV1 frozen-control representation;
- mature legacy action vocabulary;
- external-sampling Deep CFR;
- separate 3H and HU brains;
- sampled AveragePolicy trajectories;
- 2,000,000-sample reservoir capacity per memory per domain;
- 600 roots per iteration;
- 100 Advantage optimizer steps per domain per iteration;
- batch size 1024;
- 4000 AveragePolicy optimizer steps per domain at milestone finalization;
- 31 root workers, one worker numerical thread, 8 parent Torch threads, vectorized batching, production concurrent-fit mode.

## Preserved milestones

LT1: 1.2M roots, SHA256 `beef9bee9439de8d9153450d190e62b0388a678b0778c4185108361f626b4337`.

LT2 Stage A: 1.8M roots / iteration 3000, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

LT2 Stage B: 4.5M roots / iteration 7500, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Stage B final sample state:

- 3H roots 2,452,500;
- HU roots 2,047,500;
- 3H Advantage seen 76,144,669;
- HU Advantage seen 62,622,782;
- 3H AveragePolicy seen 5,549,800;
- HU AveragePolicy seen 2,072,704.

Resource gate passed with zero swap and minimum observed WSL MemAvailable 7.473 GiB.

## Statistical correction

The earlier 1000-scenario weak-baseline review was useful as a pilot, not as a decisive gate. Its Stage-B raw chip-EV confidence intervals were wide: approximate 95% half-widths ranged from about 9 to 20 chips/hand across baseline/domain cells. Apparent HU negatives against passive caller and jammer were therefore not resolved.

Similarly, checkpoint cross-play is seed-sensitive. A 3000-scenario run mildly favored Stage A; an independent 9000-scenario run reversed all primary signs to mildly favor Stage B. Neither established a strength ordering.

Policy drift, however, is clearly nontrivial: mean TV 0.041395, p95 0.108574, argmax disagreement 12.10%, with stronger HU postflop movement. Thus training is changing the policy, but existing strength estimators have not told us whether the movement is useful.

## Immediate 30k weak-baseline gate

The next experiment evaluates both preserved checkpoints against the transparent weak curriculum opponents using six independent 5000-scenario seed blocks.

Primary Stage-B claims:

- uniform legal 3H and HU;
- passive caller 3H and HU;
- jammer 3H and HU.

The six claims use a Bonferroni simultaneous family-wise 95% confidence interval. There is no arbitrary strength score. A cell is positive only if its simultaneous lower bound is above zero, negative only if its upper bound is below zero, otherwise unresolved.

Why 30k: using the observed pilot variance, the worst cell would require about 29.4k total scenarios to target an approximately 5-chip/hand simultaneous half-width. The 5-chip figure is a precision target selected to resolve the pilot's apparent ~8 to ~12 chip HU losses, not a pass threshold.

Launcher:

`tools/run_lt2_weak_baseline_variance_review.sh`

Analysis:

`tools/analyze_lt2_weak_baseline_multiseed.py`

## Training-dynamics hypotheses to test only if needed

If weak-baseline strength is negative or near-zero after adequate precision, do not merely add roots. Run bounded diagnostics first.

The first hypothesis is Advantage-network fit sufficiency. The trainer resets each domain's Advantage network every iteration and then trains it for 100 optimizer steps from the accumulated reservoir. This follows the current Deep-CFR design, but whether 100 steps are enough at a 2M reservoir is an empirical question. Measure held-out loss for the canonical 100-step fit versus larger controlled budgets before changing anything.

The second hypothesis is AveragePolicy approximation. Finalization trains the policy network for 4000 optimizer steps over the 2M policy reservoir. Measure held-out strategy loss/calibration and whether longer fitting materially changes policy quality before assuming the learned average strategy itself is correct.

Other bounded checks:

- reservoir age/composition and iteration weighting;
- 3H/HU-specific fit differences;
- blind/street concentration of errors;
- sensitivity to network capacity only after fit-budget sufficiency is known.

## DeepCrusher placement

DeepCrusher is deferred. It is a sophisticated advanced rules strategy and should be used later, after SpinCore has demonstrated statistically stable superiority over weak transparent opponents.

A literal C++ translation is not logically required for future DeepCrusher benchmarking. It may be useful for speed, auditability, or exact structural comparison, but any future benchmark can use another execution path if that path is shown faithful to the original strategy and relevant OpenPPL/library semantics.

## Decision branches

If all six weak-baseline cells are clearly positive with adequate precision, consider another bounded continuation from Stage B, then repeat the same statistically defined curriculum gate.

If any cell is clearly negative, run the training-dynamics audit before more roots.

If cells remain unresolved but the precision target is met, the edge is practically close enough to zero that training dynamics should still be investigated before more long compute.

Do not use DeepCrusher as a pass/fail requirement for this stage.

## Immediate direction

1. Preserve Stage A and Stage B.
2. Keep training stopped at iteration 7500.
3. Run `bash tools/run_lt2_weak_baseline_variance_review.sh`.
4. Review the 30k multi-seed report.
5. Only then decide between bounded continuation and training-dynamics diagnostics.
6. DeepCrusher remains later, not the current dependency.
