# SpinCore Current Work

Date: 2026-09-17
Status: **LT2 STAGE B PASS — 4.5M ROOTS — POLICY MOVED, STRENGTH STILL UNRESOLVED — 30K MULTI-SEED WEAK-BASELINE VARIANCE GATE NEXT**

## Active source of truth

Read before new compute:

- `docs/LT2_VARIANCE_AND_WEAK_BASELINE_GATE_20260917.md`
- `docs/LT2_CHECKPOINT_CROSSPLAY_REVIEW_20260917.md`
- `docs/LT2_POLICY_DRIFT_REVIEW_20260917.md`
- `docs/LT2_STAGE_B_LEARNING_REVIEW_20260917.md`
- `docs/LT2_STAGE_B_RESOURCE_REVIEW_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`

Preserve all completed checkpoints. Do not continue training beyond iteration 7500 until the variance-first weak-baseline gate is reviewed.

## Correction of immediate direction

The previous instruction to make a faithful DeepCrusher benchmark the immediate next gate was premature.

DeepCrusher is a sophisticated hard-coded strategy and remains a **later advanced benchmark**, not the correct early curriculum gate for a SpinCore policy whose performance against uniform/passive/jammer opponents is not yet established with adequate statistical precision.

A literal C++ DeepCrusher transcription is also **not a prerequisite** for ever using DeepCrusher as a reference. It is one possible faithful execution/audit route. No DeepCrusher work is required for the current SpinCore diagnosis.

## Preserved checkpoints

Stage A:

- iteration 3000 / 1.8M roots;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B:

- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Stage B has all four 2M memories in replacement regime. Resource status is healthy: no swap and minimum observed WSL MemAvailable 7.473 GiB.

## What is known

### Weak-baseline pilot was underpowered

The Stage-B 1000-scenario weak-baseline evaluation had wide raw chip-EV confidence intervals. Approximate 95% half-widths ranged from about 9 to 20 chips/hand. In HU the apparent passive/jammer values were roughly -8 and -12 chips/hand, but their intervals crossed zero widely.

Therefore those apparent losses cannot be treated as established losses.

### Stage B policy did move materially

Stage A -> Stage B policy drift on 11,040 identical decision states:

- mean TV 0.041395;
- p95 TV 0.108574;
- argmax disagreement 12.10%;
- much larger movement postflop/HU.

Training is not distributionally stagnant.

### Checkpoint cross-play is seed-sensitive

3000-scenario seed 20260918 primary B-A result was mildly negative; independent 9000-scenario seed 20260919 reversed all primary signs to mildly positive. Both runs' intervals included zero.

A proper inverse-variance combination of the two primary ALL estimates is near zero, and the two ALL/3H runs show sign instability rather than a reproducible ordering. Row-level inspection of the 9000-scenario run shows only ~4.19% of paired seat-runs had non-zero terminal B-A delta.

Conclusion: cross-play did not establish Stage-B improvement or regression.

## Immediate mathematically grounded gate

Run:

`tools/run_lt2_weak_baseline_variance_review.sh`

Design:

- six independent seed blocks: 20260920..20260925;
- 5000 scenarios per seed;
- 30,000 total scenarios **per checkpoint**;
- Stage A and Stage B evaluated on identical scenario/deal/opponent/hero RNG streams within each seed;
- no training;
- 31 workers by default.

Primary question: does Stage B have positive raw chip EV against each of `UNIFORM_LEGAL`, `PASSIVE_CALLER`, and `JAMMER`, separately in 3H and HU?

There are six primary claims. They use a Bonferroni simultaneous family-wise 95% confidence interval. Classification is mathematical rather than threshold-chosen:

- `POSITIVE` if simultaneous lower bound > 0;
- `NEGATIVE` if simultaneous upper bound < 0;
- otherwise `UNRESOLVED`.

The run size comes from the observed pilot variance. About 29.4k scenarios are required to target a worst-case simultaneous CI half-width of roughly 5 chips/hand if variance remains similar; 30k is therefore the bounded design. The 5-chip value is a **precision target**, chosen to resolve the pilot's apparent ~8 to ~12 chip HU losses; it is not a strength pass score.

## What happens next

If all six Stage-B weak-baseline claims are clearly positive, the early curriculum gate is passed and a bounded continuation can be considered.

If any claim is clearly negative, or if the result is near zero with the precision target met, do not add blind roots. Move directly to a training-dynamics audit. High-priority hypotheses include the Advantage network being reset every iteration and then fitted for only 100 optimizer steps, and whether 4000 AveragePolicy finalization steps adequately fit the 2M policy reservoir. These are hypotheses to test on held-out memory, not conclusions yet.

DeepCrusher is deferred until SpinCore is demonstrably strong and stable against these weak opponents.

## Immediate user action

Pull current `main` and run:

```bash
bash tools/run_lt2_weak_baseline_variance_review.sh
```

Wait for `LT2_WEAK_BASELINE_VARIANCE_REVIEW_PASS`, then send `SpinCore_LT2_weak_baseline_multiseed_30k.json` from Windows Downloads. Do not resume training first.
