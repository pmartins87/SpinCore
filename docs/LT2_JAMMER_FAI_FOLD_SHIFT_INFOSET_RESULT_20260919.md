# SpinCore — LT2 Jammer FAI fold-shift low-noise infoset result

Date: 2026-09-19
Status: **PASS / INCONCLUSIVE PRIMARY GATE — DIRECTION ALIGNS WITH FULL-POPULATION RESULT BUT 48 ANCHORS PER SHIFT GROUP DO NOT RESOLVE INFOSET POLICY VALUE**

## Integrity

Report schema:

`SPINCORE_LT2_JAMMER_FAI_FOLD_SHIFT_INFOSET_V1`.

The audit used:

- forensic seeds `20260920..20260925`;
- 48 `B_MORE_FOLD` anchors;
- 48 `B_LESS_FOLD` anchors;
- 64 uniform compatible opponent hands × 8 future boards per anchor;
- 512 explicit reference deals/anchor;
- no training roots;
- no optimizer steps;
- no training-memory writes;
- holdout `20261001..20261006` untouched.

Maximum Stage-A/B canonical action-gap discrepancy:

`2.9802322387695312e-08`.

The common-reference invariance contract passed.

The candidate pools exactly reproduce the full-population fold-shift counts:

- B_MORE_FOLD: 7,374;
- B_LESS_FOLD: 3,287;
- NO_FOLD_SHIFT: 5,618.

## Primary result

### B_MORE_FOLD

Stage-B-minus-A infoset policy value:

- mean `-5.92295` chips;
- seed-cluster CI95 `[-19.60209,+7.75619]`.

Stage-B-minus-A policy regret:

- mean `+5.92295`;
- CI95 `[-7.75619,+19.60209]`.

Fold-mass shift:

- mean `+0.48293`;
- seed-cluster CI95 `[+0.36245,+0.60342]`.

Reference FOLD-minus-CONTINUE margin:

- mean `-30.34896` chips;
- seed-cluster CI95 `[-67.27457,+6.57666]`.

Optimal-class counts:

- CONTINUE: 28;
- FOLD: 20.

The direction is consistent with harmful overfolding, but the primary CI crosses zero.

### B_LESS_FOLD

Stage-B-minus-A infoset policy value:

- mean `+7.47850` chips;
- seed-cluster CI95 `[-7.99668,+22.95369]`.

Stage-B-minus-A policy regret:

- mean `-7.47850`;
- CI95 `[-22.95369,+7.99668]`.

Fold-mass shift:

- mean `-0.44757`;
- seed-cluster CI95 `[-0.52362,-0.37151]`.

Reference FOLD-minus-CONTINUE margin:

- mean `-27.12024` chips;
- seed-cluster CI95 `[-59.50915,+5.26868]`.

Optimal-class counts:

- CONTINUE: 25;
- FOLD: 23.

The control direction is beneficial, matching the full-population sign, but is also unresolved.

## MSE / class metrics

No resolved Stage-B degradation appears in canonical action-gap MSE or class-error mass in either shift group.

B_LESS_FOLD raw-target MSE improves:

- B-A `-0.00061502`;
- seed-cluster CI95 `[-0.00106495,-0.00016509]`.

This is not the primary causal metric and does not by itself resolve the fold-policy question.

## Interpretation

The full-population actual-deal result remains valid:

- B_MORE_FOLD is harmful in the natural evaluation population;
- B_LESS_FOLD is beneficial.

This 96-anchor low-noise infoset audit does **not** falsify that mechanism.

Its mean directions match the population result:

- more fold -> negative;
- less fold -> positive.

But the sample is highly heterogeneous and 48 anchors/group are insufficient to resolve the infoset-level policy-value effect.

Therefore neither of the original terminal branches is justified:

- do not declare the infoset overfold mechanism confirmed yet;
- do not declare it neutral/positive and abandon the mechanism.

The correct verdict is **underpowered / inconclusive**.

## Pre-existing structural focus

Before this audit, the full-population reconciliation had already localized most of the loss to:

- legal actions `{FOLD,CALL}`;
- one public action before FAI.

A secondary decomposition of the 96 current anchors using those already-established structural criteria gives, for B_MORE_FOLD ∩ legal `{0,1}` ∩ path length 1:

- 28 anchors;
- seed-mean policy-value B-A approximately `-15.34` chips;
- seed-cluster CI95 approximately `[-29.84,-0.85]`.

This is promising but is not promoted to the final gate because that structural subset was not the primary aggregate of the current audit.

## Next gate

Run a powered, pre-registered structural infoset confirmation on exactly the high-impact structure already identified upstream:

- common Jammer FAI state;
- one public action before FAI;
- legal actions exactly `{FOLD,CALL}`;
- classify only by Stage-B-minus-A fold-mass sign;
- 32 B_MORE_FOLD anchors per seed;
- 32 B_LESS_FOLD anchors per seed;
- 384 anchors total;
- same 64 hands × 8 boards low-noise reference.

Selection still does not use:

- sampled FAI action;
- terminal outcome;
- actual-deal Q;
- low-noise Q;
- optimal class.

No training is authorized.
