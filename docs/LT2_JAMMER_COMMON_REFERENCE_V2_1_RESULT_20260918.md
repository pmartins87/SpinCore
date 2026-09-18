# SpinCore — Jammer FACING_ALL_IN common-reference V2.1 result

Date: 2026-09-18
Status: **PASS — ACTION-GAP INVARIANCE CONFIRMED; K4 STRONGLY IMPROVES THE ESTIMATOR ON FORENSIC STATES; CAUSAL EVIDENCE IS MIXED, SO NO TRAINING YET**

## Source integrity

Read-only audit on 24 actual Jammer FACING_ALL_IN first-divergence states:

- 4 anchors per already-seen forensic seed;
- seeds `20260920..20260925`;
- holdout family `20261001..20261006` untouched;
- no training roots;
- no optimizer steps;
- Stage A / Stage B checkpoints unchanged.

Expected pass marker observed:

`LT2_JAMMER_FACING_ALLIN_COMMON_REFERENCE_V2_1_PASS`.

## Action-gap invariant — PASS

For the same explicit opponent hand and future board, Stage A and Stage B raw Deep-CFR targets may differ because:

`target[a] = Q(a) - V_sigma`.

Their current traverser policies differ, so `V_sigma` differs.

V2.1 removed that arbitrary scalar by using:

`Q(a) - mean_legal(Q)`.

Observed:

- maximum Stage-A/B canonical fixed-deal target difference:
  `2.98e-08`;
- maximum raw common-offset magnitude:
  `0.416667`.

Thus the fixed-deal **action-value geometry is stage invariant to numerical precision**. The prior V2 failure was indeed an over-strong raw-target assertion, not a solver inconsistency.

## Common-reference aggregate result

On the 24 selected Jammer FACING_ALL_IN divergence states:

### Stage A

- AveragePolicy TV to common reference: `0.3069`;
- current Advantage-policy TV: `0.4543`.

### Stage B

- AveragePolicy TV to common reference: `0.3119`;
- current Advantage-policy TV: `0.4266`.

### Stage-B minus Stage-A

- AveragePolicy TV: `+0.0049`;
- AveragePolicy regret: `+0.45` chips;
- Advantage-policy TV: `-0.0277`;
- Advantage-policy regret: `+16.73` chips.

The selected divergence states themselves had mean forensic terminal B-A delta `-26.21` chips.

## K4 vs K1 on the exact same forensic states

K4 minus K1:

- target MSE: `-0.026189`;
- policy-TV diagnostic: `-0.1146`;
- reference-best-action regret: `-19.27` chips.

This is strong evidence that future-board averaging materially improves the **target estimator** on the actual Jammer-facing failure states.

## Important interpretation caveat: TV is gauge-sensitive

The common reference is expressed in the deliberate gauge:

`Q(a)-mean_legal(Q)`.

Action-value differences, expected value, best action and regret are invariant to adding a scalar to every legal action.

Regret-matching policy TV is **not** invariant to such a scalar shift, because the positive-regret set changes with the chosen zero.

Therefore:

- target MSE in the fixed gauge is useful;
- best-action/value regret is a primary gauge-invariant metric;
- AveragePolicy / Advantage expected value under the reference is meaningful;
- TV to the canonical regret-matching policy is retained as a diagnostic only and must not drive the causal decision by itself.

## Full-JSON statistical review

The full report shows:

### Stage-B minus Stage-A

AveragePolicy regret:
- mean `+0.454` chips;
- 95% CI `[-2.447,+3.355]`.

Advantage-policy regret:
- mean `+16.734` chips;
- 95% CI `[-6.365,+39.833]`.

Neither Stage-B degradation is statistically resolved on this 24-anchor sample.

### K4 minus K1

Target MSE:
- mean `-0.026189`;
- 95% CI `[-0.030418,-0.021961]`.

Reference-best-action regret:
- mean `-19.267` chips;
- 95% CI `[-30.066,-8.469]`.

Thus the K4 estimator improvement **is** statistically resolved.

### Mean action mass

Common reference:
- FOLD `0.2500`;
- CHECK_CALL `0.4792`;
- ALL_IN `0.2708`.

Stage A AveragePolicy:
- FOLD `0.3120`;
- CHECK_CALL `0.3923`;
- ALL_IN `0.2957`.

Stage B AveragePolicy:
- FOLD `0.2830`;
- CHECK_CALL `0.4101`;
- ALL_IN `0.3069`.

The mean Stage-B AveragePolicy action mass actually moves closer to the common reference on FOLD/CHECK_CALL at the aggregate level. This reinforces that the local deployed-policy degradation is not cleanly demonstrated by the 24-anchor summary.

## Outcome-equivalence correction

The sampled transition counts were:

- `0->1`: 11;
- `1->0`: 2;
- `1->9`: 7;
- `9->1`: 4.

All 11 CHECK_CALL<->ALL_IN transitions had:
- identical reference values for actions 1 and 9;
- forensic terminal B-A delta exactly `0`.

After an opponent jam, those two actions are benchmark-outcome-equivalent:

`CONTINUE = {CHECK_CALL, ALL_IN}`.

Therefore **11/24 = 45.8%** of the selected anchors were not behaviorally responsible for the Jammer EV regression.

On the 13 outcome-relevant FOLD-vs-CONTINUE anchors:

- K4-minus-K1 MSE approximately `-0.02560`, 95% CI `[-0.03178,-0.01943]`;
- K4-minus-K1 regret approximately `-16.77` chips, 95% CI `[-34.54,+1.00]`;
- Stage-B-minus-Stage-A Advantage regret approximately `+29.17`, CI `[-11.86,+70.20]`;
- AveragePolicy regret approximately `+0.71`, CI `[-3.92,+5.33]`.

The estimator MSE benefit remains resolved, but the value/regret causal chain is not.

## What V2.1 establishes

V2.1 establishes that K4 is a genuinely better **target estimator** on these states.

It does **not** establish that target variance is the cause of the deployed-policy regression strongly enough to justify training.

The pre-declared gate requiring coherent Stage-B AveragePolicy and Advantage degradation is therefore **not met**.

## Decision

**Do not train K4.**

The next experiment moves away from Jammer-specific drilling.

Run a cross-street future-chance audit on:
- outcome-relevant Jammer preflop FOLD-vs-CONTINUE states;
- PassiveCaller FLOP regression states;
- UniformLegal TURN regression states;
- matched-context controls.

If the future-chance mechanism appears across streets, prefer a general future-chance estimator intervention over a HU-preflop/Jammer-specific patch.

Canonical next contract:

- `docs/LT2_CROSS_STREET_FUTURE_CHANCE_AUDIT_20260918.md`.

The untouched holdout seeds remain sealed.
