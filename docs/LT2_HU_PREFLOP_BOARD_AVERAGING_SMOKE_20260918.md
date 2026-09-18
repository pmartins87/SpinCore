# SpinCore — LT2 HU preflop board-averaging mechanics smoke

Date: 2026-09-18
Status: **COMPLETE — PASS**

Canonical result:

- `docs/LT2_HU_PREFLOP_BOARD_AVERAGING_SMOKE_RESULT_20260918.md`

The corrected K1-vs-K4 mechanics smoke passed all invariants.

Measured on the same 8 deterministic HU roots:

- samples per arm: `273`;
- preflop samples: `22`;
- preflop targets changed under K4: `21/22 = 95.45%`;
- postflop samples: `251`;
- postflop non-identical targets: `0`;
- K1 nodes: `1,229`;
- K4 nodes: `4,765`;
- K4/K1 node multiplier: `3.8771x`;
- maximum preflop target absolute delta: `0.592`.

Verified:
- same root jobs;
- same sample count/order/identity;
- canonical postflop targets unchanged;
- preflop targets board-averaged;
- canonical RNG progression preserved;
- no optimizer steps;
- no training-memory writes;
- Stage-B checkpoint unchanged.

The first-run RNG-coupling failure and its fix remain documented in:

- `docs/LT2_HU_PREFLOP_BOARD_AVERAGING_SMOKE_FAILURE_20260918.md`.

A mechanics PASS is necessary but not sufficient to train K4.

Next gate:

- `docs/LT2_STAGE_A_B_FIRST_DIVERGENCE_FORENSIC_20260918.md`
- launcher: `tools/run_lt2_stage_a_b_first_divergence.sh`

Do not start a K4 training continuation before the deployed-policy causal attribution is reviewed.
