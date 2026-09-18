# SpinCore Current Work

Date: 2026-09-18
Status: **LT2 STAGE B PASS — 4.5M ROOTS — HU-JAMMER NEGATIVE — BOARD-ONLY K4 MECHANICS FIRST RUN FAILED SAFELY — RNG COUPLING FIXED — RERUN SMOKE**

## Active source of truth

Read before new compute:

- `docs/LT2_HU_PREFLOP_BOARD_AVERAGING_SMOKE_FAILURE_20260918.md`
- `docs/LT2_HU_PREFLOP_BOARD_AVERAGING_SMOKE_20260918.md`
- `docs/LT2_HU_PREFLOP_BOARD_ONLY_AVERAGING_RESULT_20260918.md`
- `docs/LT2_HU_PREFLOP_TARGET_ESTIMATOR_BUDGET_RESULT_20260917.md`
- `docs/LT2_HU_PREFLOP_CONDITIONAL_RESAMPLING_RESULT_20260917.md`
- `docs/LT2_WEAK_BASELINE_VARIANCE_RESULT_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`

Preserve Stage A and Stage B. Do not continue root training beyond iteration 7500.

## Preserved checkpoints

Stage A: iteration 3000 / 1.8M roots, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Confirmed practical failure

Powered 30k weak-baseline gate:

- Stage B HU Jammer raw chip EV `-5.141`, simultaneous family-wise 95% CI `[-9.078,-1.204]`;
- Stage-B-minus-Stage-A HU-Jammer paired delta `-1.682`, simultaneous six-claim interval approximately `[-3.143,-0.222]`.

Long root scaling remains paused.

## Mechanism evidence so far

HU-preflop conditional decomposition:

- future-board variance: **65.88%**;
- opponent-hand posterior variance: **26.10%**;
- residual exact-level-1 opponent-action variance: **1.74%**;
- current-model MSE to conditional mean: **6.27%**.

Board-only K4 is the measured estimator compute elbow, but this remains an estimator finding, not proof that target noise caused the Stage-A -> Stage-B HU-Jammer regression.

## First mechanics-smoke failure — resolved as implementation bug

The first K1-vs-K4 smoke stopped with:

`RuntimeError: HU board averaging changed preflop sample count across boards`.

This was a successful gate failure: no memory writes, optimizer steps, or checkpoint mutation occurred.

Root cause:

- the Advantage traversal is depth-first;
- postflop branches consume RNG before recursion returns to later preflop branches;
- different future boards therefore consume different postflop RNG histories;
- merely resetting one global RNG state at traversal start does **not** preserve later preflop opponent samples.

The implementation now records the canonical board-0 sequence of sampled **preflop opponent actions** and replays that exact sequence on alternate boards while asserting the same observation/legal set at every replayed node. Postflop remains ordinary external sampling. Board 0 remains canonical and its terminal RNG state is restored after the alternate-board work.

The generic collector gained only an overridable opponent-sampling hook. Its default remains the historical direct `sample_action` path, so canonical K1 semantics stay unchanged.

## Immediate gate — rerun mechanics smoke

Run:

```bash
bash tools/run_lt2_hu_preflop_board_averaging_smoke.sh
```

Required:

- same K1/K4 root/sample count;
- same sample order/identity;
- canonical postflop targets identical;
- at least one preflop target changed;
- K4 node multiplier measured;
- Stage-B source unchanged.

Expected marker:

`LT2_HU_PREFLOP_BOARD_AVERAGING_SMOKE_PASS`.

## Scientific gate after smoke

**Do not train K4 immediately even if the smoke passes.**

The next gate is Stage-A -> Stage-B causal attribution. It must test whether the observed regression is actually explained by target-noise/sign/policy errors in the states that changed from A to B, especially HU preflop and FACING_ALL_IN.

Only if K4 corrects the same directional errors that distinguish Stage B from Stage A will it graduate from "useful estimator" to "causally justified training intervention."

This guards against tuning to the Jammer benchmark or overfitting a convenient proxy.

DeepCrusher remains deferred.

## Immediate user action

Pull current `main` and rerun `bash tools/run_lt2_hu_preflop_board_averaging_smoke.sh`.

Wait for `LT2_HU_PREFLOP_BOARD_AVERAGING_SMOKE_PASS` or the first error. Send the output/JSON. Do not start any training continuation.
