# SpinCore — LT2 Jammer FAI structural infoset confirmation result

Date: 2026-09-19  
Status: **PASS — INFOSET-LEVEL OVERFOLD CONFIRMED IN THE PRE-REGISTERED HIGH-IMPACT FAI STRUCTURE**

## Integrity

Report schema:

`SPINCORE_LT2_JAMMER_FAI_STRUCTURAL_INFOSET_CONFIRMATION_V1`.

The powered confirmation used:

- forensic seeds `20260920..20260925`;
- common Jammer FAI states only;
- legal actions exactly `{FOLD,CALL}`;
- exactly one public action before FAI;
- 32 `B_MORE_FOLD` + 32 `B_LESS_FOLD` anchors per seed;
- 384 anchors total;
- 64 compatible opponent hands × 8 future boards = 512 explicit deals/anchor;
- 196,608 explicit reference deals;
- no training roots;
- no optimizer steps;
- no training-memory writes;
- holdout `20261001..20261006` untouched.

Maximum Stage-A/B canonical action-gap discrepancy:

`2.9802322387695312e-08`.

The common-reference invariance contract passed.

## Primary confirmatory result — B_MORE_FOLD

N = 192.

Stage-B-minus-A infoset policy value:

- mean `-24.54921` chips;
- seed-cluster CI95 `[-35.58425,-13.51416]`.

This satisfies the pre-registered primary condition with a fully negative seed-cluster interval.

Policy regret increases by the exact opposite amount:

- `+24.54921` chips;
- CI95 `[+13.51416,+35.58425]`.

Fold mass:

- B-A `+0.46907`;
- CI95 `[+0.45307,+0.48508]`.

Low-noise reference FOLD-minus-CONTINUE value:

- mean `-40.86355` chips;
- CI95 `[-57.51783,-24.20927]`.

So, in this selected structural population, CONTINUE is better on average by about 40.86 chips while Stage B adds about 46.9 percentage points of fold mass.

Optimal reference classes:

- CONTINUE: 112;
- FOLD: 80.

## Model-quality diagnostics also resolve

Unlike the earlier underpowered audit, the powered structural test resolves degradation in the fitted Advantage geometry itself.

Stage-B-minus-A canonical action-gap MSE:

- `+0.000714714`;
- seed-cluster CI95 `[+0.000397781,+0.001031647]`.

Stage-B-minus-A class-error mass:

- `+0.127198`;
- seed-cluster CI95 `[+0.058510,+0.195887]`.

Raw-target MSE B-A remains unresolved:

- `+0.000121541`;
- CI95 `[-0.000303594,+0.000546677]`.

This is important: aggregate raw-target MSE does not reveal the harmful fold-vs-continue action-gap drift.

Fallback incidence:

- Stage A: `6.25%`;
- Stage B: `50.00%`.

Stage-A policy regret:

- `30.90899` chips.

Stage-B policy regret:

- `55.45820` chips.

## Directional control — B_LESS_FOLD

N = 192.

Policy value B-A:

- `+3.05309` chips;
- CI95 `[-5.04231,+11.14848]`.

The sign is beneficial but the interval remains unresolved.

Fold mass B-A:

- `-0.36531`;
- CI95 `[-0.41129,-0.31932]`.

Reference FOLD-minus-CONTINUE:

- `-15.64822` chips;
- CI95 `[-27.30691,-3.98952]`.

The control therefore remains directionally compatible with the population result, but the primary confirmation does not depend on it.

## Verdict

The Stage-B Jammer FAI overfold is now confirmed at the decision-time infoset level inside the high-impact structure that had been localized before this test.

This rules out the explanation that the full-population loss was merely covariance with the realized hidden hand or future board.

However, the same report shows that a fallback-only explanation would be too narrow:

- fallback incidence rises sharply;
- canonical action-gap MSE also worsens significantly;
- class-error mass worsens significantly;
- raw-target MSE does not resolve.

The next diagnostic must separate:

1. common raw-output offset / zero-crossing / fallback effects;
2. fold-vs-continue action-gap drift;
3. nonlinear interaction between those two mechanisms.

No training change is authorized yet.
