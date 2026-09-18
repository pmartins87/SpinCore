# SpinCore — Long-Training Plan

Status: **LT2 STAGE B PASS — ROOT TRAINING PAUSED — CROSS-STREET FUTURE-CHANCE AUDIT PASS — GENERAL K4 CAUSAL CASE NOT SUPPORTED — FULL JSON REVIEW PENDING**
Date: 2026-09-18

## Preserved milestones

Stage A:
- iteration 3000 / 1.8M roots;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Never rewrite either checkpoint.

## Confirmed regressions

Jammer:
- B-A `-1.682`, CI `[-2.767,-0.597]`;
- FACING_ALL_IN contribution `-1.126`, CI `[-2.049,-0.202]`.

PassiveCaller:
- B-A `-1.261`, CI `[-2.377,-0.145]`;
- FLOP contribution `-0.672`, CI `[-1.299,-0.046]`.

UniformLegal:
- overall unresolved;
- TURN subgroup `-0.552`, CI `[-0.970,-0.134]`.

Long scaling remains frozen.

## Existing target-estimator evidence

HU preflop:
- future-board variance **65.88%**;
- opponent-hand variance **26.10%**;
- opponent-action residual **1.74%**;
- model error to conditional mean **6.27%**.

Exact1 is rejected on compute efficiency.

Board-only K4:
- mechanics PASS;
- estimator benefit confirmed.

## V2.1 full statistical review

On 24 Jammer FACING_ALL_IN selected divergence states:

K4 minus K1:
- target MSE `-0.026189`, 95% CI `[-0.030418,-0.021961]`;
- best-action regret `-19.27`, CI `[-30.07,-8.47]`.

Stage B minus Stage A:
- AveragePolicy regret `+0.454`, CI `[-2.447,+3.355]`;
- Advantage regret `+16.734`, CI `[-6.365,+39.833]`.

The estimator improvement is resolved.
The Stage-B policy/model degradation on the selected sample is not.

## Outcome-equivalence correction

V2.1 transitions:
- 11 `0->1`;
- 2 `1->0`;
- 7 `1->9`;
- 4 `9->1`.

All 11 CHECK_CALL<->ALL_IN transitions had equal benchmark reference values and zero realized B-A terminal delta.

Thus 45.8% of V2.1 anchors were benchmark-neutral.

On the 13 FOLD-vs-CONTINUE anchors:
- K4 MSE benefit remains resolved;
- K4 regret benefit is unresolved;
- Stage-B Advantage and AveragePolicy degradation are unresolved.

The narrow K4 causal gate therefore fails.

## Cross-street future-chance result

The 72-anchor read-only audit passed.

Terminal summary:

### Jammer preflop
FAILURE:
- future-board `0.692`;
- model `0.308`;
- K4-K1 MSE `-0.030440`;
- regret `-36.58`.

CONTROL:
- future-board `0.820`;
- model `0.180`;
- K4-K1 MSE `-0.054250`;
- regret `-10.28`.

### PassiveCaller flop
FAILURE:
- future-board `0.610`;
- action `0.212`;
- model `0.179`;
- K4-K1 MSE `-0.012697`;
- regret `-11.52`.

CONTROL:
- future-board `0.350`;
- action `0.230`;
- model `0.421`;
- K4-K1 MSE `-0.012868`;
- regret `-4.51`.

### UniformLegal turn
FAILURE:
- future-board `0.143`;
- action `0.070`;
- model `0.787`;
- K4-K1 MSE `-0.008399`;
- regret `-19.67`.

CONTROL:
- future-board `0.333`;
- action `0.088`;
- model `0.579`;
- K4-K1 MSE `-0.004301`;
- regret `-19.51`.

## Interpretation

Future-chance averaging is a genuine estimator improvement, but high future-board noise is not specific to failing states.

The turn result is particularly important: the failure sample is model-error dominated, which points away from chance averaging as the main explanation.

A generalized all-street K4 intervention is not authorized.

Before designing the next mechanism experiment, inspect the full JSON confidence intervals and per-anchor distributions.

## Immediate direction

1. Keep Stage A/B frozen.
2. Upload `SpinCore_LT2_cross_street_future_chance.json`.
3. Review FAILURE-vs-CONTROL uncertainty and per-anchor structure.
4. Keep holdout seeds `20261001..20261006` untouched.
5. Do not train K4 or resume long training.
6. Do not launch another diagnostic until that review is complete.
