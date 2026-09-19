# SpinCore — LT2 Jammer FAI broad action-gap / regret-matching calibration

Date: 2026-09-19
Status: **ACTIVE — TEST WHETHER SMALL ADVANTAGE SIGN/RANKING ERRORS ARE AMPLIFIED BY PRODUCTION REGRET MATCHING**

## Trigger

The current-behavior first-divergence forensic resolved:

- Jammer total BEH B-A `-8.4304`, CI95 `[-12.5155,-4.3453]`;
- `PREFLOP_FACING_ALL_IN` contribution `-6.2267`, CI95 `[-9.2575,-3.1959]`;
- FAI explains **73.86%** of total Jammer current-behavior regression;
- root contribution is `-2.2037`, but unresolved;
- no Jammer postflop first divergence.

Prior work also showed:
- Jammer FAI low-noise target is Stage-A/B stationary;
- Stage-B aggregate own-target MSE is not clearly worse;
- future-board K4 improves estimator quality but is not failure-specific.

Therefore aggregate MSE is likely hiding a policy-sensitive error mode.

## Hypothesis

Production behavior uses `lean_regret_matching_policy`:

1. if any legal Advantage output is positive:
   - clip negatives to zero;
   - normalize positive values;

2. if every legal Advantage output is non-positive:
   - use softmax fallback over raw outputs.

This creates nonlinear sensitivity to:
- zero crossings;
- which actions enter positive support;
- relative positive-regret scale;
- all-nonpositive fallback regime.

A small MSE change can therefore cause a large behavior-policy change.

## Broad selection

Do not select states because A and B diverge at FAI.

Use all current-behavior paired Jammer trajectories that:

- reach preflop immediately after opponent ALL_IN;
- remain on a common Stage-A/B public trajectory until that state.

Record the anchor **before** sampling the FAI hero action.

Then deterministically sample 8 anchors per forensic seed = 48 total.

The selection is independent of:
- FAI A/B sampled action divergence;
- terminal B-A outcome.

This is the anti-overfitting control.

## Common reference

At FAI:
- Jammer behavior is hand-independent;
- opponent is already all-in.

Reference:

- 32 uniformly stratified compatible opponent hands;
- 8 future boards per hand;
- 256 explicit deals/anchor;
- exact0 is sufficient after opponent all-in;
- canonical action-gap gauge:
  `Q(a)-mean_legal(Q)`.

Stage-A/B fixed-deal action-gap invariance remains asserted on a validation subset.

## Stage-specific true Advantage target

For Stage `S`, let production model policy be `sigma_S`.

Using the common Q-like reference:

`A*_S(a) = Q(a) - sum_b sigma_S(b) Q(b)`.

This is the correct stage-specific zero point for comparing the raw Advantage model to its own current behavior target.

## Primary metrics

Per stage and paired B-A:

- raw-target MSE;
- canonical action-gap MSE;
- expected policy regret in chips;
- best-action agreement;
- exact positive-support agreement;
- positive-support Jaccard;
- false-positive action count;
- false-negative action count;
- all-nonpositive fallback incidence;
- policy mass on truly negative actions;
- FOLD and non-FOLD policy mass;
- class-error mass;
- best-vs-second reference gap;
- non-FOLD reference-value spread.

Primary inferential intervals are also clustered by forensic seed.

## Decision logic

If Stage B has similar MSE but:
- more support/sign mistakes;
- more fallback use;
- more mass on truly negative actions;
- or materially higher policy regret;

then the causal defect is **regret-matching calibration / ranking**, not target variance.

Possible intervention would then target:
- loss weighting near zero crossings;
- ranking/action-gap supervision;
- support-aware calibration;
- fallback semantics;

but only after the exact failure mode is resolved.

If Stage B is not worse on broad FAI calibration despite the paired EV loss, investigate trajectory weighting / root interaction instead.

## Launcher

```bash
bash tools/run_lt2_jammer_fai_broad_calibration.sh
```

Expected marker:

`LT2_JAMMER_FAI_BROAD_CALIBRATION_PASS`.

Holdout seeds remain untouched.

No training is authorized.
