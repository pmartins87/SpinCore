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
- Board-only averaging sweep — **COMPLETE; K4 IS ESTIMATOR COMPUTE ELBOW**.
- Board-averaging mechanics smoke — **PASS; IMPLEMENTATION ISOLATED CORRECTLY**.
- Stage-A -> Stage-B deployed-policy forensic — **FIRST RUN TERMINATED BY RESOURCE PRESSURE; LIGHTWEIGHT-POLICY FIX READY; RERUN NEXT**.
- K4 causal training pilot — **NOT AUTHORIZED YET**.
- Root training beyond iteration 7500 — **PAUSED**.
- DeepCrusher — **DEFERRED**.

Canonical current files:

- `CURRENT_WORK.md`
- `docs/LT2_HU_PREFLOP_BOARD_AVERAGING_SMOKE_RESULT_20260918.md`
- `docs/LT2_STAGE_A_B_FIRST_DIVERGENCE_FORENSIC_20260918.md`
- `docs/LT2_STAGE_A_B_FIRST_DIVERGENCE_RESOURCE_FAILURE_20260918.md`
- `docs/LT2_HU_PREFLOP_BOARD_ONLY_AVERAGING_RESULT_20260918.md`
- `docs/LT2_HU_PREFLOP_TARGET_ESTIMATOR_BUDGET_RESULT_20260917.md`
- `docs/LT2_WEAK_BASELINE_VARIANCE_RESULT_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`

## Preserved checkpoints

Stage A: iteration 3000 / 1.8M roots, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Keep both unchanged.

## Why long scaling remains frozen

Stage B HU Jammer remains confirmed negative:
- raw EV `-5.141` chips/hand;
- simultaneous family-wise 95% CI `[-9.078,-1.204]`.

The successful K4 mechanics smoke proves only implementation correctness.

It does not prove causal responsibility for the strength regression.

## K4 mechanics result

On 8 paired prospective HU roots:

- K1/K4 sample counts identical: 273;
- 22 preflop samples;
- 21/22 preflop targets changed under K4;
- 251 postflop samples;
- zero postflop target differences;
- K1 nodes `1,229`;
- K4 nodes `4,765`;
- multiplier `3.8771x`.

The intervention is isolated to preflop labels as designed.

## Anti-overfitting rule

Do not promote K4 merely because:
- HU Jammer is the failing benchmark;
- board averaging improves a target proxy;
- the implementation now works.

First identify the actual Stage-A -> Stage-B deployed-policy regression.

The diagnostic weak-baseline seed family `20260920..20260925` may be used for forensic localization because it is already seen.

It must not be reused for future candidate acceptance.

Reserve `20261001..20261006` as untouched holdout seeds for any later acceptance gate.

## Forensic resource correction

The first attempt used 31 spawned workers and loaded the two complete training checkpoints inside every worker. That needlessly replicated reservoirs and both domains and ended in an abrupt `Terminated` shutdown before any report was produced.

The rerun extracts the HU AveragePolicy once per source checkpoint in the parent and gives workers only lightweight policy snapshots. Default concurrency is now 16. The forensic statistical design is unchanged.

## Immediate forensic gate

Launcher:

`tools/run_lt2_stage_a_b_first_divergence.sh`

Scope:
- TRUE_HEADS_UP only;
- all three weak baselines;
- Stage A versus Stage B stored AveragePolicy;
- same scenario/deal/seat/random streams until first sampled hero-action divergence;
- mutually exclusive first-divergence groups:
  - NO_DIVERGENCE;
  - PREFLOP_ROOT;
  - PREFLOP_FACING_ALL_IN;
  - PREFLOP_OTHER;
  - FLOP;
  - TURN;
  - RIVER.

Each group's B-minus-A chip delta contributes additively to total paired B-minus-A EV.

## Branch after forensic audit

If the negative contribution is postflop, deprioritize K4.

If the regression is preflop but not in K4-sensitive contexts, K4 remains unproven.

If HU-preflop/FACING_ALL_IN carries a resolved negative contribution and Stage B shifts mass in the same direction implicated by lower-variance target diagnostics, run a final matched Advantage/target overlay.

Only after that overlay may a bounded K4 training pilot be considered.

Jammer-only localization is not sufficient by itself; cross-baseline consistency strengthens a general causal interpretation.

## Immediate action

Run `bash tools/run_lt2_stage_a_b_first_divergence.sh`, stop at PASS or first error, and keep root training paused.
