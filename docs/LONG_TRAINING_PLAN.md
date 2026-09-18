# SpinCore — Long-Training Plan

Status: **LT2 STAGE B PASS — ROOT TRAINING PAUSED AT 4.5M — K4 MECHANICS PASS — FORENSIC RESOURCE FIX READY — RERUN ACTIVE**
Date: 2026-09-18

## Current state

The line has reached:

- LT0: 120k roots;
- LT1: 1.2M roots;
- LT2 Stage A: 1.8M roots;
- LT2 Stage B: 4.5M roots / iteration 7500;
- powered weak-baseline gate confirms Stage B HU Jammer negative;
- HU-preflop target variance is dominated by hidden/chance variation;
- exact1 is not compute-efficient;
- board-only future-board averaging has a measured K4 estimator elbow;
- the corrected K4 mechanics smoke passed;
- no K4 training continuation is authorized yet;
- immediate next step is direct Stage-A -> Stage-B AveragePolicy regression attribution.

Read first:

- `LT2_HU_PREFLOP_BOARD_AVERAGING_SMOKE_RESULT_20260918.md`
- `LT2_STAGE_A_B_FIRST_DIVERGENCE_FORENSIC_20260918.md`
- `LT2_STAGE_A_B_FIRST_DIVERGENCE_RESOURCE_FAILURE_20260918.md`
- `LT2_HU_PREFLOP_BOARD_ONLY_AVERAGING_RESULT_20260918.md`

## Preserved milestones

Stage A: iteration 3000 / 1.8M roots, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Never rewrite either preserved checkpoint.

## Strength gate

Stage B HU Jammer is `-5.141` chips/hand with simultaneous family-wise 95% CI `[-9.078,-1.204]`.

Stage B minus Stage A HU Jammer is `-1.682`, simultaneous six-claim interval approximately `[-3.143,-0.222]`.

Blind root scaling remains disallowed.

## Target-estimator evidence

HU-preflop conditional decomposition:

- future-board variance: **65.88%**;
- opponent-hand posterior variance: **26.10%**;
- residual exact-level-1 opponent-action variance: **1.74%**;
- current-model MSE to conditional mean: **6.27%**.

Board-only K4 reduces raw target noise and improves branch mismatch/regret enough to be a plausible estimator change.

Exact1 remains rejected as the next intervention.

## K4 implementation result

The corrected smoke verified the experimental K4 path without any training:

- same 8 deterministic HU roots;
- 273 samples in K1 and K4;
- 22 preflop samples;
- 21 preflop targets changed;
- 251 postflop samples;
- zero postflop target differences;
- K1 nodes 1,229;
- K4 nodes 4,765;
- node multiplier 3.8771x;
- canonical RNG progression preserved;
- source checkpoint unchanged.

This proves implementation isolation only.

## Causal chain that still must be demonstrated

The deployed evaluator uses the stored **AveragePolicy**.

K4 changes the Advantage target estimator.

The intended causal chain would be:

`lower-variance Advantage targets -> better Advantage behavior -> better sampled strategy data -> better AveragePolicy -> better chip EV`.

We have evidence for the first arrow only.

Before training, first locate the actual A->B AveragePolicy regression.

## Forensic first-run resource failure

The first full attempt terminated abruptly with 31 spawned workers. Each worker loaded both complete training checkpoints, including large reservoirs and both domains, even though only the HU AveragePolicy is required. This is a resource-design failure, not strategy evidence.

The corrected audit extracts policy-only HU snapshots once in the parent, releases the full checkpoints, and gives workers only those small snapshots. Default concurrency is 16. No scientific semantics changed.

## Active Stage-A -> Stage-B forensic

Use the already-seen diagnostic seed family:

`20260920..20260925`

Default 5,000 full-sampler scenarios per seed.

Retain HU scenarios and evaluate all three weak baselines.

For each scenario/deal/opponent/hero seat:

- Stage A and Stage B use identical random streams;
- states are identical until the sampled hero action first differs;
- every seat-run is assigned to:
  - NO_DIVERGENCE;
  - PREFLOP_ROOT;
  - PREFLOP_FACING_ALL_IN;
  - PREFLOP_OTHER;
  - FLOP;
  - TURN;
  - RIVER;
- the terminal B-minus-A chip delta is attributed to that first-divergence group;
- group contributions must add back to total paired B-minus-A chip EV.

At divergence also measure:
- Stage-A -> Stage-B action transition;
- AveragePolicy TV;
- B-A probability-mass shift in FOLD/CHECK_CALL/ALL_IN.

## Seed firewall

The forensic seeds are already seen and are now design data.

They must not be used to accept a future intervention.

Reserve the untouched family:

`20261001..20261006`

for any future candidate acceptance gate.

Do not inspect those holdout seeds before an intervention is frozen.

## Decision branches

If the negative A->B contribution is mainly postflop, HU-preflop K4 is not the primary fix.

If it is preflop but not in the contexts that board averaging addresses, K4 remains unproven.

If HU-preflop/FACING_ALL_IN carries a resolved negative contribution and the Stage-B action-mass shift matches the lower-variance target pathology, run one final Advantage/target overlay on matched states.

Only after that overlay may a bounded K4 training pilot be admitted.

Jammer-only evidence is insufficient; a general mechanism should have at least supporting cross-baseline evidence or a baseline-independent target explanation.

## DeepCrusher placement

DeepCrusher remains deferred.

## Immediate direction

1. Preserve Stage A and Stage B.
2. Keep training stopped at iteration 7500.
3. Run `bash tools/run_lt2_stage_a_b_first_divergence.sh`.
4. Review `SpinCore_LT2_stage_a_b_first_divergence.json`.
5. Do not train K4 before the forensic result is reviewed.
6. Keep `20261001..20261006` untouched.
