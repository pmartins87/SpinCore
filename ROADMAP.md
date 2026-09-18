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
- Cross-street future-chance audit — **PASS; BOARD NOISE GENERALIZES BUT DOES NOT EXPLAIN FAILURES BY ITSELF**.
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

## Cross-street result

Terminal-level evidence:

- Jammer preflop: future-board variance large in both FAILURE and CONTROL; CONTROL is higher.
- PassiveCaller flop: FAILURE future-board fraction `0.610` vs CONTROL `0.350`; plausible association, but K4 MSE gain is almost identical.
- UniformLegal turn: FAILURE is model-error dominated at `0.787`; future-board fraction only `0.143`; K4 regret gain nearly identical to CONTROL.

Thus future-chance averaging is a useful estimator improvement but is **not yet the general causal explanation** for Stage-B regression.

## Next gate

Review the full `SpinCore_LT2_cross_street_future_chance.json` before defining another experiment.

Primary questions:
- are FAILURE-vs-CONTROL component differences statistically resolved?
- is flop future-board excess robust?
- is turn model-error excess robust?
- do per-anchor patterns point to target nonstationarity, function-approximation drift, or coverage/forgetting?

No training is authorized.

Holdout `20261001..20261006` remains sealed.

## Immediate action

Upload `SpinCore_LT2_cross_street_future_chance.json`.

Do not run another diagnostic or any training yet.
