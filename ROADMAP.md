# SpinCore Roadmap — active state 2026-09-18

## Active status

- LT0 — **DONE**.
- LT1 — **DONE**.
- LT2 Stage A — **PASS**.
- LT2 Stage B — **PASS** at 4.5M roots / iteration 7500.
- Weak-baseline regression — **CONFIRMED**.
- AveragePolicy extra-fit — **NOT SUPPORTED**.
- Advantage optimizer escalation — **NOT SUPPORTED**.
- HU-preflop target variance — **HIDDEN/CHANCE DOMINANT**.
- K4 board averaging — **VALID ESTIMATOR IMPROVEMENT**.
- Jammer K4 causal gate — **NOT MET**.
- Cross-street future-chance audit — **PASS; FUTURE-CHANCE NOT FAILURE-SPECIFIC**.
- Full cross-street JSON review — **COMPLETE**.
- Stage-A/B target-drift and model-tracking matrix — **NEXT**.
- K4 training — **NOT AUTHORIZED**.
- long root training — **PAUSED**.
- DeepCrusher — **DEFERRED**.

## Preserved checkpoints

Stage A SHA:
`e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B SHA:
`3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Cross-street full-JSON conclusion

Future-board noise is real but not a failure discriminator.

- Jammer controls are significantly noisier than failures in absolute future-board variance.
- Passive flop has a different fractional composition, but no resolved absolute future-board excess.
- Turn has higher model-error fraction in failures, but no resolved absolute model-error excess.
- K4 MSE improvement does not significantly separate FAILURE from CONTROL.

Therefore the next question is not "where else should K4 be enabled?"

It is:

**Did the self-play target move from Stage A to B, or did the model fail to track its own moving target?**

## Next gate

`tools/run_lt2_cross_street_target_drift_tracking.sh`

Same three contexts:
1. Jammer preflop FOLD-vs-CONTINUE;
2. PassiveCaller FLOP;
3. UniformLegal TURN.

Same FAILURE/CONTROL selection.

For each anchor:
- same hidden deal;
- same future boards;
- same target RNG seeds;
- Stage-A conditional target;
- Stage-B conditional target;
- Stage-A current Advantage model;
- Stage-B current Advantage model;
- canonical action-gap comparison.

## Decision after target-drift matrix

- high target drift + models fit own targets -> self-play nonstationarity;
- low target drift + Stage-B own-target error increase -> approximation/coverage/forgetting;
- high drift + high tracking error -> both;
- neither -> search outside the Advantage target/model chain.

No training before this distinction is measured.

Holdout `20261001..20261006` remains sealed.

## Immediate action

Run `bash tools/run_lt2_cross_street_target_drift_tracking.sh`.

Stop at PASS or first error. Do not train.
