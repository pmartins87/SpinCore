# SpinCore Current Work

Date: 2026-09-21
Status: **NATIVE C++ INFERENCE PARITY PASS — OPENHOLDEM CANONICAL-STATE RECONSTRUCTION PREREQUISITE NEXT**

## Frozen strategic/runtime identities

Python hybrid deployment:
`87e46b40cb43bb89cb46bf3b760bbac5c8282491fd3d6da73d1bbe28329b278c`

Native C++ deployment:
`2b79ab7ff746a9c1c3dd73dbc0a1d6884a471813cf34b9cb126790c4c4cbb123`

Checkpoint:
`a51dbbed71090e45f2c4f5db6297f72eab848990b38650702b436e2aa60ca4bf`

HU ensemble:
`c44b817f75304db352eedc33febbd180e6d038e0580c91be0b9befdbdb08f181`

## Native inference parity

PASS:
- 7,302 decision records;
- 5,242 3H;
- 2,060 HU;
- all four streets covered;
- max probability drift `2.527e-05` under `2e-4` tolerance;
- argmax mismatches 0;
- illegal mass 0;
- nonfinite outputs 0.

The neural runtime is no longer the blocker.

## OpenHoldem reconstruction problem

Real OpenHoldem does not know opponent private cards or future board cards, but the authoritative solver constructor accepts a complete deal.

The runtime plan is:
- preserve Hero hole cards;
- preserve visible board;
- fill only unknown opponent holes and unrevealed future board deterministically;
- replay the exact observed public action path;
- obtain canonical SPNNIV1/legal/action semantics from the authoritative solver.

Before using this architecture, hidden fillers must be proven irrelevant to current public-state inference.

## Immediate action

```bash
bash tools/run_lt2_runtime_hidden_filler_invariance.sh
```

Wait for `LT2_RUNTIME_HIDDEN_FILLER_INVARIANCE_PASS`, then send
`SpinCore_LT2_runtime_hidden_filler_invariance.json`.

No training and no holdout reuse.
