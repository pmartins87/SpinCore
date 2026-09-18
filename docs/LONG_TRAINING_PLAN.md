# SpinCore — Long-Training Plan

Status: **LT2 STAGE B PASS — ROOT TRAINING PAUSED — K4 ESTIMATOR BENEFIT REAL BUT NARROW CAUSAL GATE NOT MET — CROSS-STREET VARIANCE AUDIT ACTIVE**
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

## Scientific priority

Do not continue narrowing around Jammer.

Test whether the **future-chance target-noise mechanism generalizes across streets**.

Canonical contract:

`docs/LT2_CROSS_STREET_FUTURE_CHANCE_AUDIT_20260918.md`.

Groups:
- Jammer preflop FOLD-vs-CONTINUE;
- PassiveCaller FLOP;
- UniformLegal TURN.

For every group:
- FAILURE states;
- matched-context CONTROL states;
- actual hidden opponent hand fixed;
- visible board fixed;
- only unrevealed future cards resampled.

Reference:
- 8 boards × 4 repeats;
- exact1.

Candidate:
- 8 disjoint boards;
- exact0;
- K1 vs K4.

Primary questions:
- how much sample-target MSE is future-board variance?
- is that fraction higher in FAILURE than CONTROL?
- does K4 reduce MSE?
- does K4 reduce gauge-invariant best-action regret?

## Decision logic

If the same effect appears across preflop/flop/turn, build a generalized future-chance estimator intervention.

If only Jammer preflop shows it, keep K4 as a local candidate only.

If postflop is dominated by model error or opponent-action variance, investigate those mechanisms instead.

No training resumes until this mechanism decision is made.

## Immediate direction

1. Keep Stage A/B frozen.
2. Run `bash tools/run_lt2_cross_street_future_chance.sh`.
3. Wait for `LT2_CROSS_STREET_FUTURE_CHANCE_AUDIT_PASS`.
4. Send `SpinCore_LT2_cross_street_future_chance.json`.
5. Keep holdout seeds `20261001..20261006` untouched.
6. Do not train K4 or resume long training.
