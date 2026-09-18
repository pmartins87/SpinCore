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

## What V2.1 supports

V2.1 supports two concrete statements:

1. **Stage B current Advantage is worse in value/regret on these selected Jammer-facing states** despite having slightly lower TV to the arbitrary canonical-RM policy.
2. **K4 substantially improves target estimation on these same selected states**, including a large `-19.27` chip reduction in candidate-policy regret.

This closes much of the local mechanism chain for the Jammer-facing failure.

## What V2.1 does not establish

The AveragePolicy degradation on these selected states is small in the console aggregate:

- TV `+0.0049`;
- regret `+0.45` chips.

The full JSON confidence intervals and action-mass shifts must be inspected before declaring this link resolved.

Also, the global Stage-B regression is still multi-mechanism:

- PassiveCaller has a resolved FLOP regression;
- UniformLegal has a resolved TURN subgroup.

A HU-preflop-only K4 patch may therefore solve one symptom while leaving the broader training-instability mechanism untouched.

## Decision

**Do not train K4 yet.**

Next:

1. inspect the full V2.1 JSON, especially 95% intervals and FOLD/CHECK_CALL/ALL_IN mass shifts;
2. treat gauge-invariant regret/value metrics as primary;
3. decide whether the next experiment should be:
   - a bounded K4 pilot, if the local causal chain is statistically coherent; or
   - a cross-street variance/regression audit, if evidence suggests a broader target-estimator instability that also explains FLOP/TURN regressions.

The untouched holdout seeds remain sealed.
