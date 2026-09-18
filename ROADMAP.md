# SpinCore Roadmap — active state 2026-09-18

## Active status

- LT0 calibration — **DONE**: 120k roots.
- LT1 production-shaped milestone — **DONE**: 1.2M roots.
- LT1 physical fit optimization — **PASS**.
- LT2 Stage A — **PASS**: 1.8M roots.
- LT2 Stage B — **PASS**: 4.5M roots / iteration 7500.
- Stage B resource gate — **PASS**.
- Policy drift Stage A -> Stage B — **MATERIAL MOVEMENT CONFIRMED**.
- Checkpoint cross-play — **NO REPRODUCIBLE ORDERING**.
- 30k weak-baseline gate — **COMPLETE; HU JAMMER NEGATIVE**.
- AveragePolicy extra-budget hypothesis — **NOT SUPPORTED**.
- Advantage optimizer escalation — **NOT SUPPORTED AS NEXT INTERVENTION**.
- HU-preflop conditional variance decomposition — **COMPLETE; 93.73% HIDDEN/CHANCE VARIANCE**.
- Exact0/exact1 target-estimator sweep — **COMPLETE; EXACT0 + MORE DEALS WINS COMPUTE FRONTIER**.
- Board-only averaging sweep — **COMPLETE; K4 ADMITTED AS COMPUTE ELBOW**.
- Board-averaging mechanics smoke — **NEXT**.
- Bounded K4 causal training pilot — **PENDING SMOKE**.
- Root training beyond iteration 7500 — **PAUSED**.
- DeepCrusher — **DEFERRED**.

Canonical current files:

- `CURRENT_WORK.md`
- `docs/LT2_HU_PREFLOP_BOARD_ONLY_AVERAGING_RESULT_20260918.md`
- `docs/LT2_HU_PREFLOP_BOARD_AVERAGING_SMOKE_20260918.md`
- `docs/LT2_HU_PREFLOP_TARGET_ESTIMATOR_BUDGET_RESULT_20260917.md`
- `docs/LT2_WEAK_BASELINE_VARIANCE_RESULT_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`

## Preserved checkpoints

Stage A: iteration 3000 / 1.8M roots, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Keep both unchanged.

## Strength failure freezing scale

Stage B HU Jammer remains confirmed negative:
- raw EV `-5.141` chips/hand;
- simultaneous family-wise 95% CI `[-9.078,-1.204]`.

No long continuation until a bounded semantic intervention beats preserved Stage B.

## Mechanism and estimator decisions

The dominant HU-preflop target noise is hidden/chance variance:
- future board **65.88%**;
- opponent hand **26.10%**;
- residual exact-level-1 opponent-action noise **1.74%**.

Exact1 is rejected as the next intervention because its ~2.1x same-K node cost does not produce a matched-compute policy benefit.

Board-only averaging is operationally much simpler and directly attacks the largest component.

## Board-only result

Overall K1 -> K4:
- MSE `0.042638 -> 0.013556`;
- policy TV `0.5001 -> 0.4868`;
- regret `38.73 -> 35.42`;
- branch mismatch `26.04% -> 17.68%`;
- nodes `75.5 -> 302.1`.

K4 -> K8:
- doubles nodes;
- materially lowers raw MSE;
- does not resolve further TV/regret/argmax improvement;
- only a small branch-mismatch gain remains.

Therefore **K4 is admitted; K8 is not**.

In FACING_ALL_IN, K1 -> K8 produces a resolved TV improvement and K4 is already essentially at the K8 policy-space plateau.

## Implementation principle

The experiment must alter only HU-preflop Advantage labels:
- same scenario/root distribution;
- same hole cards;
- canonical board retained as board 0;
- K-1 alternate future boards conditional on fixed holes;
- same external-sampling RNG replayed for all boards;
- preflop targets averaged;
- postflop targets retained from canonical board only;
- same sample count/order/identity;
- canonical K1 remains default.

## Immediate gate

Run:

```bash
bash tools/run_lt2_hu_preflop_board_averaging_smoke.sh
```

The smoke is read-only and compares identical HU roots at K1 and K4.

It must verify:
- sample identity invariance;
- postflop target invariance;
- actual preflop target change;
- node multiplier;
- source checkpoint integrity.

After smoke review:
1. use measured K4/K1 node multiplier to set a bounded root budget;
2. run one isolated K4 continuation candidate from preserved Stage B;
3. compare candidate vs Stage B using the powered weak-baseline suite;
4. only reopen long training if the candidate passes.

## Immediate action

Wait for `LT2_HU_PREFLOP_BOARD_AVERAGING_SMOKE_PASS` after running the smoke launcher. Do not train first.
