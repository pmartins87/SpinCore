# SpinCore Roadmap — active state 2026-09-18

## Active status

- LT0 — **DONE**.
- LT1 — **DONE**.
- LT2 Stage A — **PASS**: 1.8M roots.
- LT2 Stage B — **PASS**: 4.5M roots / iteration 7500.
- Weak-baseline gate — **HU JAMMER NEGATIVE; PASSIVE HU REGRESSION CONFIRMED**.
- AveragePolicy extra-fit — **NOT SUPPORTED**.
- Advantage optimizer escalation — **NOT SUPPORTED**.
- HU-preflop conditional variance — **HIDDEN/CHANCE DOMINANT**.
- exact0/exact1 estimator sweep — **EXACT1 NOT COMPUTE-EFFICIENT**.
- board-only K4 — **ESTIMATOR BENEFIT CONFIRMED**.
- K4 mechanics smoke — **PASS**.
- Stage-A -> Stage-B forensic — **COMPLETE**.
- Jammer common-reference V2.1 — **PASS; K4 ESTIMATOR IMPROVES, BUT POLICY/MODEL CAUSAL GATE UNRESOLVED**.
- Outcome-equivalence review — **45.8% OF V2.1 ANCHORS WERE CALL<->ALLIN BENCHMARK-NEUTRAL**.
- Cross-street future-chance audit — **NEXT**.
- K4 training pilot — **NOT AUTHORIZED**.
- long root training — **PAUSED**.
- DeepCrusher — **DEFERRED**.

## Preserved checkpoints

Stage A SHA:
`e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B SHA:
`3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## V2.1 statistical verdict

K4 improves estimator quality on the 24 selected Jammer-facing states:
- MSE delta `-0.026189`, CI entirely negative;
- regret delta `-19.27` chips, CI entirely negative.

But Stage-B degradation relative to Stage A is unresolved on those anchors:
- AveragePolicy regret `+0.45`, CI crosses zero;
- Advantage regret `+16.73`, CI crosses zero.

On the 13 outcome-relevant FOLD-vs-CONTINUE anchors, K4 MSE remains resolved, but K4 regret and Stage-B policy/model degradation are not.

Therefore a Jammer-specific K4 training pilot would be premature.

## Anti-overfitting direction

We now test the mechanism across all resolved regression contexts instead of drilling the same benchmark.

Cross-street groups:
1. Jammer preflop FOLD-vs-CONTINUE;
2. PassiveCaller FLOP;
3. UniformLegal TURN.

Each includes FAILURE and matched-context CONTROL states.

The audit isolates **future board chance** by keeping:
- exact dealt hidden opponent hand;
- visible board prefix;
- public path;
- actor;
- observation;
- legal actions

fixed while resampling only unrevealed future cards.

## Decision after cross-street audit

If future-board variance and K4 estimator benefit appear across streets, design a generalized future-chance estimator by street.

If the effect is isolated to Jammer preflop, K4 remains a narrow local candidate and should not be mistaken for the root cause of Stage-B instability.

If postflop regressions are dominated by model or opponent-action noise instead, investigate those mechanisms separately.

## Immediate action

Run `bash tools/run_lt2_cross_street_future_chance.sh`.

Do not train before its result is reviewed.

Holdout `20261001..20261006` remains sealed.
