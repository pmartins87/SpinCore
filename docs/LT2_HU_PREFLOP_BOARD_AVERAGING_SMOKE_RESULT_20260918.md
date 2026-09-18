# SpinCore — LT2 HU preflop board-averaging mechanics smoke result

Date: 2026-09-18
Status: **PASS — IMPLEMENTATION CONTRACT VERIFIED; NO TRAINING AUTHORIZED**

## Source

Preserved Stage-B checkpoint:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

The corrected smoke completed with:

`LT2_HU_PREFLOP_BOARD_AVERAGING_SMOKE_PASS`.

The source checkpoint remained unchanged.

## Mechanics result

Same 8 deterministic prospective HU roots were compared under K1 and K4.

- roots: 8;
- samples per arm: 273;
- preflop samples: 22;
- preflop targets changed by K4: 21 / 22 = **95.45%**;
- postflop samples: 251;
- postflop non-identical targets: **0**;
- maximum preflop target absolute delta: `0.592`;
- K1 nodes: `1,229`;
- K4 nodes: `4,765`;
- measured node multiplier: **3.8771x**.

All contractual invariants passed:

- same root jobs;
- same sample count;
- same sample order and identity;
- canonical postflop targets unchanged;
- preflop targets board-averaged;
- canonical RNG progression preserved;
- no optimizer steps;
- no training-memory writes.

## Interpretation

The corrected implementation now does what the experiment claims mechanically.

This result says **nothing by itself about strategy strength** and does not authorize a K4 continuation.

The earlier estimator studies established that K4 lowers HU-preflop target variance. The smoke establishes that the training implementation isolates that intervention without contaminating sample identity or postflop labels.

It does **not** establish that future-board target noise caused the Stage-A -> Stage-B HU-Jammer regression.

## Scientific next gate

Before any K4 training, perform direct Stage-A -> Stage-B deployed-policy attribution.

The weak-baseline evaluator plays the stored **AveragePolicy**, not the Advantage network. Therefore the first causal question is:

**Where, in the actual paired evaluation trajectories, does the AveragePolicy behavior of Stage B first diverge from Stage A and where does the negative B-A chip-EV contribution occur?**

The next audit reuses the already-seen diagnostic seed family from the powered weak-baseline gate and all three weak opponent families. It does not train or tune anything.

Future candidate acceptance must use a fresh, untouched seed family rather than these forensic seeds.
