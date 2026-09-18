# SpinCore — LT2 cross-street future-chance audit result

Date: 2026-09-18
Status: **PASS — FUTURE-CHANCE NOISE GENERALIZES TO FLOP, BUT DOES NOT EXPLAIN THE WHOLE REGRESSION; FULL JSON REVIEW REQUIRED BEFORE NEXT EXPERIMENT**

## Integrity

Read-only cross-street audit completed:

- 72 anchors;
- forensic seeds `20260920..20260925`;
- holdout `20261001..20261006` untouched;
- Stage A/B checkpoints unchanged;
- no training roots;
- no optimizer steps.

Expected marker:

`LT2_CROSS_STREET_FUTURE_CHANCE_AUDIT_PASS`.

## Terminal summary

### JAMMER preflop FOLD-vs-CONTINUE

FAILURE:
- future-board fraction `0.692`;
- opponent-action fraction `0.000`;
- model fraction `0.308`;
- K4-K1 MSE `-0.030440`;
- K4-K1 regret `-36.58`.

CONTROL:
- future-board fraction `0.820`;
- opponent-action fraction `0.000`;
- model fraction `0.180`;
- K4-K1 MSE `-0.054250`;
- K4-K1 regret `-10.28`.

Interpretation: future-board noise is large, but it is **not failure-specific**. Controls have even more board-variance fraction and larger MSE benefit from K4.

### PASSIVE_CALLER flop

FAILURE:
- future-board fraction `0.610`;
- opponent-action fraction `0.212`;
- model fraction `0.179`;
- K4-K1 MSE `-0.012697`;
- K4-K1 regret `-11.52`.

CONTROL:
- future-board fraction `0.350`;
- opponent-action fraction `0.230`;
- model fraction `0.421`;
- K4-K1 MSE `-0.012868`;
- K4-K1 regret `-4.51`.

Interpretation: the flop failure states have a much larger **relative** future-board component than controls, so future chance remains a plausible contributor. However the absolute K4 MSE improvement is almost identical in failure and control samples. The full confidence intervals are required before attributing the flop regression to this mechanism.

### UNIFORM_LEGAL turn

FAILURE:
- future-board fraction `0.143`;
- opponent-action fraction `0.070`;
- model fraction `0.787`;
- K4-K1 MSE `-0.008399`;
- K4-K1 regret `-19.67`.

CONTROL:
- future-board fraction `0.333`;
- opponent-action fraction `0.088`;
- model fraction `0.579`;
- K4-K1 MSE `-0.004301`;
- K4-K1 regret `-19.51`.

Interpretation: the turn failure sample is **model-error dominated**, not future-chance dominated. K4 regret improvement is nearly identical in failure and control. Future-board averaging therefore does not currently explain the resolved turn regression.

## Provisional scientific verdict

The broad hypothesis:

> Stage-B regression across streets is primarily caused by future-board target variance.

is **not supported by the terminal summary**.

What is supported:

1. future-board averaging is a genuine estimator-quality improvement in preflop and flop;
2. high future-board variance exists in many controls too, so variance alone is not sufficient to identify failing states;
3. flop failures may have a higher relative future-board contribution than controls;
4. turn failures are primarily model-error dominated.

Therefore a generalized all-street K4 training intervention is **not authorized**.

The next decision should be based on the full JSON:
- confidence intervals for every FAILURE/CONTROL decomposition;
- K4-K1 MSE/regret intervals;
- per-anchor model-error distributions;
- whether flop FAILURE vs CONTROL differences are statistically meaningful;
- whether turn model error is systematically elevated in failure states.

## Next action

Upload:

`SpinCore_LT2_cross_street_future_chance.json`

Do not run new training or a new diagnostic before the full JSON review.
