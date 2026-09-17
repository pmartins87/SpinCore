# SpinCore — LT2 variance-first weak-baseline gate

Date: 2026-09-17
Status: **ACTIVE — DEEPCRUSHER DEFERRED; WEAK-BASELINE STRENGTH/VARIANCE MUST BE RESOLVED FIRST**

## Correction of direction

DeepCrusher is a sophisticated hard-coded strategy and is not an appropriate immediate pass/fail gate for the current early SpinCore policy. It remains a later advanced external benchmark.

A literal C++ transcription of DeepCrusher is also not a prerequisite for ever using DeepCrusher as a reference. It is one possible execution/audit route. A future benchmark could use any implementation path that is demonstrably faithful to the source semantics. The immediate SpinCore work must not depend on completing that separate project.

## What the current evidence actually says

Stage B (4.5M roots) materially changed the AveragePolicy relative to Stage A (1.8M roots), especially postflop/HU. Therefore the training line is not distributionally stagnant.

However, the existing strength diagnostics have insufficient precision to support a strong conclusion:

- the original weak-baseline review used only 1000 scenarios;
- Stage-B raw chip-EV 95% half-widths were roughly 9 to 20 chips/hand depending on baseline/domain;
- apparent HU losses versus passive/jammer were about -8 to -12 chips/hand, which are not resolved by intervals this wide;
- two independent Stage-A-vs-Stage-B cross-play runs changed sign, showing that the relative checkpoint estimate is seed-sensitive.

The 9000-scenario cross-play row data also showed that only about 4.19% of paired seat-runs produced a non-zero terminal B-minus-A delta. That sparse-difference design is useful for variance reduction when trajectories stay coupled, but the remaining rare divergent trajectories carry large outcomes and make checkpoint ordering noisy.

Thus the correct statement is not “Stage B is bad” and not “Stage B is good.” The current evidence is too noisy to decide.

## Mathematical sample-size rationale

The next gate is built from the observed pilot variance rather than an arbitrary score threshold.

Primary question:

> Does Stage B have positive mean raw chip EV against each transparent weak baseline in both THREE_HANDED and TRUE_HEADS_UP?

There are six primary claims: three baselines x two domains.

The original 1000-scenario Stage-B pilot had a worst 95% half-width of about 20.14 chips/hand (HU vs uniform legal). Using the observed standard error and the usual 1/sqrt(N) scaling, a family-wise simultaneous 95% interval across six claims needs about 29.4k total scenarios to bring the worst expected half-width to about 5 chips/hand if variance remains comparable.

Therefore the new review uses **30,000 scenarios per checkpoint**, split into six independent 5,000-scenario seed blocks.

The 5-chip number is a **precision target, not a pass threshold**. It is chosen because it is tight enough to resolve whether the pilot's apparent -8 to -12 chip HU losses persist. Strength classification itself is based only on whether the simultaneous confidence interval is above, below, or includes zero.

## Multiple-testing control

For Stage B, the six primary 3H/HU weak-baseline claims use a Bonferroni simultaneous family-wise 95% normal confidence interval.

Classification per claim:

- `POSITIVE`: simultaneous lower bound > 0;
- `NEGATIVE`: simultaneous upper bound < 0;
- `UNRESOLVED`: interval includes 0.

No arbitrary EV score is used as a PASS/FAIL cutoff.

Stage A is evaluated on the exact same scenario/deal/opponent/hero RNG streams inside each seed block. Stage-B-minus-Stage-A paired deltas are secondary diagnostics and help determine whether the extra 2.7M roots improved or degraded weak-baseline performance.

## Why six independent seed blocks

A single large seed can hide seed/sampler sensitivity. Six independent blocks provide:

- a pooled high-power estimate;
- per-seed point estimates for sign stability;
- exact A/B pairing inside each block;
- direct evidence about whether the prior sign changes were sampling variance.

## Immediate run

Use:

```bash
bash tools/run_lt2_weak_baseline_variance_review.sh
```

The launcher:

- preserves both checkpoints;
- exports inference-only policies;
- evaluates Stage A and Stage B on six seeds (`20260920` through `20260925`);
- uses 5000 scenarios per seed = 30,000 scenarios per checkpoint;
- uses 31 evaluation workers by default;
- performs no training;
- creates `SpinCore_LT2_weak_baseline_multiseed_30k.json` in Windows Downloads.

## Decision after the result

If Stage B is clearly positive against all six weak-baseline/domain claims with adequate precision, the curriculum gate is passed and we can consider another bounded training continuation while still tracking learning quality.

If any weak baseline is clearly negative, do not add blind roots. Investigate the training dynamics first, especially:

- Advantage-network reset plus only 100 optimizer steps per iteration;
- held-out Advantage-memory fit quality at 100 vs larger step budgets;
- AveragePolicy fit sufficiency after 4000 final optimizer steps;
- policy-target/replay composition and reservoir age;
- 3H/HU-specific failure concentration.

If claims remain unresolved but the 5-chip precision target is met, the true edge is small enough that the next step should be a training-dynamics audit rather than simply increasing evaluation forever.

DeepCrusher remains later on the roadmap, after SpinCore demonstrates strong and statistically stable performance against these weak curriculum opponents.
