# SpinCore — LT2 cross-street future-chance audit — full JSON review

Date: 2026-09-18
Status: **COMPLETE — FUTURE-CHANCE IS A REAL ESTIMATOR ISSUE, BUT FAILURE/CONTROL DISCRIMINATION IS NOT SUPPORTED; TARGET-DRIFT / MODEL-TRACKING AUDIT NEXT**

## Integrity

The full `SPINCORE_LT2_CROSS_STREET_FUTURE_CHANCE_AUDIT_V1` report contains:

- 72 anchors;
- 12 FAILURE + 12 CONTROL anchors in each of three contexts;
- forensic seeds `20260920..20260925`;
- holdout `20261001..20261006` untouched;
- read-only execution;
- no optimizer steps;
- no training-memory writes;
- no new training roots.

Stage A and Stage B source checkpoints remained unchanged.

## Important statistical correction

The terminal summary reported **fractions** of sampled-target MSE attributable to future board, opponent action and model error.

A high fraction does not imply that the corresponding **absolute error component** is larger.

The full JSON therefore requires both:
- component fractions;
- absolute component magnitudes.

Independent FAILURE and CONTROL samples are not treated as paired.

For the comparisons below, the approximate 95% interval for the independent mean difference is:

`(mean_F - mean_C) ± 1.96 * sqrt(SEM_F^2 + SEM_C^2)`.

## JAMMER preflop FOLD-vs-CONTINUE

### Fractional composition

Future-board fraction:
- FAILURE `0.6917`;
- CONTROL `0.8198`;
- F-C `-0.1281`;
- approximate 95% CI `[-0.3140,+0.0579]`.

No resolved failure-specific excess.

### Absolute components

Absolute future-board variance:
- FAILURE `0.03834`;
- CONTROL `0.07201`;
- F-C `-0.03367`;
- approximate 95% CI `[-0.06195,-0.00538]`.

Total sampled-target MSE:
- FAILURE `0.04842`;
- CONTROL `0.09058`;
- F-C `-0.04216`;
- approximate 95% CI `[-0.07567,-0.00865]`.

The **controls are noisier than failures in absolute terms**.

### K4 discrimination

K4-K1 target MSE:
- FAILURE `-0.03044`;
- CONTROL `-0.05425`;
- failure-minus-control `+0.02381`;
- approximate 95% CI `[-0.00463,+0.05224]`.

K4-K1 regret:
- FAILURE `-36.58`;
- CONTROL `-10.28`;
- difference unresolved.

Conclusion:

**future-board noise is not a causal discriminator for the Jammer failure states.**

K4 improves the estimator, but the failing states are not the noisiest states.

## PASSIVE_CALLER flop

### Fractional composition

Future-board fraction:
- FAILURE `0.6097`;
- CONTROL `0.3497`;
- F-C `+0.2600`;
- approximate 95% CI `[+0.0456,+0.4743]`.

Model fraction:
- FAILURE `0.1788`;
- CONTROL `0.4207`;
- F-C `-0.2419`;
- approximate 95% CI `[-0.4467,-0.0371]`.

The **composition** differs.

### Absolute components

Absolute future-board variance:
- FAILURE `0.00819`;
- CONTROL `0.00468`;
- F-C `+0.00351`;
- approximate 95% CI `[-0.00369,+0.01071]`.

Absolute model error:
- FAILURE `0.00137`;
- CONTROL `0.00246`;
- F-C `-0.00109`;
- approximate 95% CI `[-0.00322,+0.00104]`.

Total sampled-target MSE:
- FAILURE `0.01309`;
- CONTROL `0.01010`;
- F-C `+0.00299`;
- approximate 95% CI `[-0.00781,+0.01379]`.

None of those absolute differences is resolved.

### K4 discrimination

K4-K1 MSE:
- FAILURE `-0.012697`, CI entirely negative within FAILURE;
- CONTROL `-0.012868`, CI entirely negative within CONTROL;
- F-C difference `+0.000171`;
- approximate 95% CI `[-0.01119,+0.01153]`.

K4-K1 regret:
- FAILURE `-11.52`, own 95% CI `[-19.33,-3.71]`;
- CONTROL `-4.51`, own 95% CI `[-16.48,+7.46]`;
- F-C difference unresolved.

Conclusion:

The flop failure sample has a different **relative composition**, but there is no resolved evidence that it has more absolute future-board noise or that K4 benefits failures more than controls.

This is not enough to call future-board noise the cause of the PassiveCaller flop regression.

## UNIFORM_LEGAL turn

### Fractional composition

Future-board fraction:
- FAILURE `0.1434`;
- CONTROL `0.3334`;
- F-C `-0.1900`;
- approximate 95% CI `[-0.3772,-0.0028]`.

Model fraction:
- FAILURE `0.7866`;
- CONTROL `0.5791`;
- F-C `+0.2076`;
- approximate 95% CI `[-0.0380,+0.4532]`.

Failure states are proportionally more model-dominated.

### Absolute components

Absolute model error:
- FAILURE `0.00812`;
- CONTROL `0.00881`;
- F-C `-0.00070`;
- approximate 95% CI `[-0.01220,+0.01081]`.

Absolute future-board variance:
- FAILURE `0.00438`;
- CONTROL `0.00325`;
- difference unresolved.

Total sampled-target MSE:
- FAILURE `0.01433`;
- CONTROL `0.01249`;
- difference unresolved.

Therefore the earlier phrase "turn failure has elevated model error" would be too strong.

The correct statement is:

**turn failures are model-error dominated in composition, but their absolute model error is not higher than controls in this sample.**

### K4 discrimination

K4-K1 regret:
- FAILURE `-19.668`, own CI `[-33.55,-5.79]`;
- CONTROL `-19.507`, own CI `[-33.80,-5.21]`.

The effects are essentially identical.

K4-K1 MSE:
- FAILURE `-0.008399`;
- CONTROL `-0.004301`;
- F-C difference unresolved.

Conclusion:

K4 is useful as an estimator improvement on turn, but the improvement is not failure-specific.

## Overall causal verdict

The cross-street data reject the simple causal rule:

`more future-chance noise -> Stage-B failure`.

Across all three contexts:

- K4 often reduces target-estimator error;
- the reduction also appears strongly in controls;
- FAILURE-vs-CONTROL K4 MSE differences are unresolved;
- absolute error-component excess is generally unresolved;
- in Jammer preflop, controls are actually **more** future-board noisy than failures.

Therefore K4 is best viewed as a general **variance-reduction improvement**, not as the demonstrated cause/fix for Stage-B regression.

## What the audit still cannot answer

This audit measured only the Stage-B target process.

It cannot distinguish:

1. **target nonstationarity / self-play drift**:
   the conditional target itself moved from Stage A to Stage B;

2. **function-approximation / tracking failure**:
   Stage B failed to fit its own current target;

3. **both**.

That distinction is now the highest-value question.

## Next gate: Stage-A / Stage-B target-drift and model-tracking matrix

Use the same FAILURE/CONTROL contexts and same forensic seeds.

For every anchor:

- generate the same explicit hidden deal and future-board samples;
- compute low-noise conditional target under Stage-A behavior;
- compute low-noise conditional target under Stage-B behavior;
- canonicalize legal-action Advantage vectors to a common action-gap gauge;
- compare Stage-A model to Stage-A target;
- compare Stage-B model to Stage-B target;
- compare Stage-A target to Stage-B target;
- compare model drift to target drift.

Primary quantities:

- target-drift MSE;
- Stage-A own-target model error;
- Stage-B own-target model error;
- B-minus-A own-target error;
- model-drift tracking error;
- reference best-action changes;
- FAILURE versus CONTROL differences.

Interpretation:

- high target drift + models tracking their own targets -> **nonstationary self-play target drift**;
- low target drift + Stage-B own-target error increase -> **approximation / forgetting / coverage failure**;
- high target drift + high tracking error -> **both**.

No training is authorized before that distinction is measured.
