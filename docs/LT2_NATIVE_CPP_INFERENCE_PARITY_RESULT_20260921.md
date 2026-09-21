# SpinCore — LT2 native C++ inference parity result

Date: 2026-09-21  
Status: **PASS — NATIVE C++ MODEL RUNTIME VALIDATED**

## Frozen identities

Python hybrid deployment bundle:

`87e46b40cb43bb89cb46bf3b760bbac5c8282491fd3d6da73d1bbe28329b278c`

Source checkpoint:

`a51dbbed71090e45f2c4f5db6297f72eab848990b38650702b436e2aa60ca4bf`

Source HU ensemble:

`c44b817f75304db352eedc33febbd180e6d038e0580c91be0b9befdbdb08f181`

Native C++ deployment binary:

`2b79ab7ff746a9c1c3dd73dbc0a1d6884a471813cf34b9cb126790c4c4cbb123`

Native binary size: 5,510,610 bytes.

## Parity coverage

The standalone native C++ runtime was compared against Python-generated frozen deployment fixtures:

- total decision records: `7,302`;
- THREE_HANDED: `5,242`;
- TRUE_HEADS_UP: `2,060`;
- preflop: `6,141`;
- flop: `822`;
- turn: `249`;
- river: `90`;
- probabilities compared: `73,020`.

## Numeric result

- maximum absolute probability difference: `2.52723693848e-05`;
- mean absolute probability difference: `2.75346224261e-08`;
- maximum probability-mass error: `1.78813934326e-07`;
- argmax mismatches: **0**;
- illegal-action mass failures: **0**;
- nonfinite failures: **0**.

Committed tolerance was `2e-4`; the observed worst-case probability drift is about 7.9x smaller.

## Decision

Native C++ inference parity: **PASS**.

The neural portion of the OpenHoldem runtime no longer needs Python or PyTorch.

No strategic test was run and the holdout was not reused.

## Next runtime blocker

The remaining correctness problem is not neural inference. It is reconstructing the exact canonical SpinCore public state from OpenHoldem.

The solver constructor requires a complete deal, while real OpenHoldem knows only Hero cards and currently exposed board cards. Runtime reconstruction will therefore use deterministic filler cards for hidden opponent holes and unrevealed future board cards, then replay the observed public action history.

Before implementing that bridge, prove that changing those hidden fillers cannot alter, at the current decision state:

- canonical SPNNIV1 observation;
- actor/domain;
- lean legal actions;
- exact lean action resolution.

That hidden-filler invariance gate uses only old forensic seeds.
