# SpinCore — LT2 Jammer FAI broad calibration result

Date: 2026-09-19
Status: **PASS — BROAD FAI CALIBRATION DOES NOT SHOW STAGE-B POLICY-REGRET DEGRADATION; SOME CALIBRATION METRICS IMPROVE; SUPPORT-JACCARD METRIC IS NOT CAUSALLY VALID; FULL-POPULATION RECONCILIATION NEXT**

## Integrity

Report schema:

`SPINCORE_LT2_JAMMER_FAI_BROAD_CALIBRATION_V1`.

The audit used:

- 48 broad common Jammer FAI anchors;
- 8 anchors per forensic seed;
- 32 uniform compatible opponent hands × 8 future boards per anchor;
- common `Q(a)-mean_legal(Q)` action-gap reference;
- no selection on FAI divergence or terminal B-A outcome;
- no training roots;
- no optimizer steps;
- no training-memory writes;
- holdout `20261001..20261006` untouched.

Candidate pools contained roughly 2685–2797 common FAI states per seed.

Maximum Stage-A/B fixed-deal canonical action-gap discrepancy:

`2.98e-08`.

The common-reference invariance contract passed.

## Primary result

### Expected policy regret

Stage A:
- mean `32.885` chips;
- seed-cluster CI95 `[22.073,43.697]`.

Stage B:
- mean `20.984` chips;
- seed-cluster CI95 `[11.894,30.073]`.

B-A:
- `-11.901` chips;
- seed-cluster CI95 `[-25.515,+1.713]`.

Therefore **Stage B is not broadly worse in policy regret on this 48-anchor sample**.

The direction is actually better for Stage B, but the paired difference is unresolved.

This rejects the simple hypothesis:

`Stage-B broad FAI calibration is globally worse -> Jammer FAI regression`.

## Canonical action-gap MSE

Stage A:
- `0.0020185`.

Stage B:
- `0.0014322`.

B-A:
- `-0.00058635`;
- seed-cluster CI95 `[-0.00093271,-0.00023998]`.

This is a **resolved Stage-B improvement**.

## Raw own-target MSE

Stage A:
- `0.0035857`.

Stage B:
- `0.0024866`.

B-A:
- `-0.0010991`;
- seed-cluster CI95 `[-0.0026814,+0.0004831]`.

Direction favors Stage B, unresolved.

## FOLD-vs-CONTINUE class error mass

Stage A:
- `0.38346`.

Stage B:
- `0.29914`.

B-A:
- `-0.08432`;
- seed-cluster CI95 `[-0.16662,-0.00203]`.

This is another **resolved Stage-B improvement**.

## Fold mass

Stage A:
- `0.27466`.

Stage B:
- `0.37475`.

B-A:
- `+0.10009`;
- seed-cluster CI95 `[+0.04693,+0.15326]`.

Stage B folds about 10 percentage points more often on the broad sampled FAI population.

That shift is real, but by itself it is not evidence of worse play because the reference sample contains both:

- 20 FOLD-optimal anchors;
- 28 CONTINUE-optimal anchors.

The class-error metric already accounts for which class is correct and improves for Stage B.

## Fallback incidence

All-nonpositive fallback:

- Stage A: `5/48 = 10.42%`;
- Stage B: `12/48 = 25.00%`.

The fallback is more frequent in Stage B.

However, because Stage-B class error and action-gap MSE improve, fallback frequency alone cannot be treated as the cause of the Jammer loss.

## Best-action match

- Stage A: `29/48 = 60.42%`;
- Stage B: `24/48 = 50.00%`.

This raw metric is also not decisive because at many FAI states CALL and ALL_IN have identical reference value.

The broad audit reports zero non-FOLD reference-value spread, so exact universal-slot argmax matching over-penalizes strategically equivalent CALL-vs-ALL_IN choices.

## Important correction: positive-support metric

The `positive_support_exact` rate is zero for both stages and Stage B has a much lower positive-support Jaccard.

That metric must **not** be used as causal evidence.

Reason:

The "true Advantage" was defined stage-specifically as:

`A*_S(a) = Q(a) - V_{sigma_S}`.

When `sigma_S` is pure on an optimal action, that selected action has true Advantage exactly zero.

But the production neural output must generally be positive on that action in order for regret matching to select it.

Therefore a strict comparison

`raw_model > 0` versus `A*_S > 0`

can label a perfectly policy-consistent optimal action as a "false positive".

The support-Jaccard metric is consequently self-referential and unsuitable as a standalone calibration verdict.

This does **not** invalidate:
- common Q reference;
- canonical action-gap MSE;
- expected policy regret;
- FOLD-vs-CONTINUE class error;
- fold-mass measurements.

## Why this conflicts with the previous first-divergence result

Previous full paired evaluation showed:

- total Jammer current-behavior B-A `-8.4304`;
- FAI additive contribution `-6.2267`, resolved.

Yet this 48-anchor low-noise broad sample does not show Stage B worse.

Possible explanations:

1. 48 anchors are too small / unrepresentative for a heterogeneous FAI population;
2. the damaging EV is concentrated in a relatively small high-leverage subset;
3. the paired first-divergence stochastic contribution needs to be reconciled against deterministic expected policy value on the exact natural population.

Before designing any intervention, resolve this apparent contradiction directly.

## Next gate: full-population actual-deal counterfactual reconciliation

Use **every** HU Jammer seat-run in the forensic seed family.

For every paired trajectory that reaches a common FAI state:

- do not subsample anchors;
- before sampling the FAI action, clone the exact solver state;
- apply every legal FAI action;
- read terminal chip delta on the already-dealt opponent hand and full board;
- compute
  `sum_a (sigma_B[a]-sigma_A[a]) Q_actual[a]`;
- also sample A/B with the exact same RNG used by the prior forensic.

The sampled contribution must reproduce the prior:

`-6.22672064777328`.

The deterministic expected contribution then answers whether the FAI loss exists on the full natural population without action-sampling noise.

This is the next causal gate.

No training is authorized.
