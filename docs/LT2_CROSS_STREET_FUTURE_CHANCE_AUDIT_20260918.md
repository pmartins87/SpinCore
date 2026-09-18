# SpinCore — LT2 cross-street future-chance causal audit

Date: 2026-09-18
Status: **ACTIVE — TEST WHETHER THE JAMMER BOARD-NOISE MECHANISM GENERALIZES TO FLOP/TURN REGRESSIONS**

## Why this gate exists

The full V2.1 JSON strengthens but does not fully close the narrow Jammer/K4 causal case.

Important full-JSON facts:

- K4 target MSE improves on all 24 selected Jammer-facing anchors.
- Aggregate K4-minus-K1:
  - target MSE `-0.026189`, 95% CI `[-0.030418,-0.021961]`;
  - reference-best-action regret `-19.27` chips, 95% CI `[-30.07,-8.47]`.
- Stage-B-minus-Stage-A:
  - AveragePolicy regret `+0.45`, 95% CI `[-2.45,+3.35]`;
  - Advantage-policy regret `+16.73`, 95% CI `[-6.36,+39.83]`.

Thus the estimator benefit is resolved, but Stage-B policy/model degradation on the 24 selected anchors is not statistically resolved.

## Outcome-equivalence correction

The V2.1 selected first-divergence transitions were:

- `0->1`: 11;
- `1->0`: 2;
- `1->9`: 7;
- `9->1`: 4.

All 11 `CHECK_CALL <-> ALL_IN` transitions had:

- identical reference values for actions 1 and 9;
- forensic terminal B-A delta exactly `0`.

After an opponent jam, those two actions belong to the same benchmark outcome class:

`CONTINUE = {CHECK_CALL, ALL_IN}`.

Therefore **45.8% of the V2.1 selected anchors were not behaviorally responsible for the Jammer EV regression**.

On the 13 outcome-relevant FOLD-vs-CONTINUE anchors:

- K4-minus-K1 MSE remains clearly negative, approximately `-0.02560`, 95% CI `[-0.03178,-0.01943]`;
- K4-minus-K1 regret is approximately `-16.77` chips, but the 95% interval approximately `[-34.54,+1.00]` crosses zero;
- Stage-B-minus-Stage-A Advantage regret is approximately `+29.17`, CI `[-11.86,+70.20]`;
- AveragePolicy regret is approximately `+0.71`, CI `[-3.92,+5.33]`.

This is not sufficient to authorize training K4.

## Strategic decision

Do **not** spend the next experiment further tuning to Jammer.

The deployed-policy forensic already found other resolved regressions:

- PASSIVE_CALLER FLOP contribution `-0.672`, CI `[-1.299,-0.046]`;
- UNIFORM_LEGAL TURN contribution `-0.552`, CI `[-0.970,-0.134]`.

The scientifically stronger next question is whether **future-chance target variance is a cross-street mechanism**.

If the same mechanism appears in flop and turn failure states, the right intervention is likely a general future-chance estimator rather than a HU-preflop/Jammer-specific patch.

## Audit groups

Use the already-seen forensic seeds only:

`20260920..20260925`.

For each seed collect FAILURE and CONTROL states in three contexts:

1. **JAMMER_PREFLOP_FACING_ALLIN_CLASS**
   - collapse CHECK_CALL and ALL_IN into CONTINUE;
   - FAILURE requires A/B FOLD-vs-CONTINUE disagreement;
   - CONTINUE-internal slot swaps are not failures.

2. **PASSIVE_CALLER_FLOP**
   - FAILURE = first raw A/B action divergence on FLOP.

3. **UNIFORM_LEGAL_TURN**
   - FAILURE = first raw A/B action divergence on TURN.

Controls are same-context states reached before a non-equivalent A/B divergence where the compared action/class agrees.

Default: 2 states per seed per status = 72 total anchors.

## What is held fixed

For every anchor:

- exact Episode;
- hero and opponent hole cards from the actually dealt hidden state;
- all already-visible board cards;
- public action path;
- actor;
- observation;
- legal action set.

Only **unrevealed future board cards** are resampled.

This deliberately isolates future chance. It does not estimate hidden-hand variance.

## Reference decomposition

Per anchor:

- 8 independent future boards;
- 4 target repeats per board;
- exact opponent level 1.

Decompose Stage-B sampled-target MSE into:

- within-board opponent-action sampling variance;
- future-board variance;
- model error to the board-conditional mean.

The balanced design must close numerically.

## Production-shaped estimator test

Use a disjoint board stream:

- exact opponent level 0;
- 8 independent future boards;
- compare K1 versus K4 board averaging.

Report:

- target MSE;
- policy-TV diagnostic;
- reference-best-action regret;
- argmax agreement;
- actual traversal nodes.

Primary metrics are MSE and value/regret.

## Decision logic

If failure states across preflop/flop/turn all show:

- substantial future-board fraction;
- K4 reducing MSE;
- and preferably K4 reducing regret;

then future-chance averaging is a **general training-estimator candidate**, and the next intervention should be generalized by street rather than hard-coded to Jammer/preflop.

If only Jammer preflop shows the effect, K4 remains a narrow local intervention.

If postflop failure states are dominated by model error or opponent-action variance instead, investigate those mechanisms separately.

The holdout family `20261001..20261006` remains untouched.

## Launcher

```bash
bash tools/run_lt2_cross_street_future_chance.sh
```

Expected marker:

`LT2_CROSS_STREET_FUTURE_CHANCE_AUDIT_PASS`

No training is authorized before this result is reviewed.
