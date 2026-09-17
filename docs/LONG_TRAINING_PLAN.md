# SpinCore — Long-Training Plan

Status: **LT2 STAGE B PASS — SAME-REGIME EXTENSION PAUSED PENDING POLICY-DRIFT DIAGNOSIS**
Date: 2026-09-17

## Current state

The continuous learning line has reached:

- LT0: 120k roots — calibration;
- LT1: 1.2M roots — production-shaped milestone;
- LT2 Stage A: 1.8M roots — first policy-reservoir saturation gate;
- LT2 Stage B: 4.5M roots / iteration 7500 — all four 2M memories saturated/replacement;
- Stage A -> Stage B paired weak-baseline review — complete, no statistically distinguishable checkpoint gain/regression;
- next gate — decision-level policy drift before any further training.

Read `LT2_STAGE_B_LEARNING_REVIEW_20260917.md` and `LT2_STAGE_B_RESOURCE_REVIEW_20260917.md`.

## Core training contract

Current functional line:

- empirical SpinGo 3H/HU/blind/stack sampling;
- WTA chip-EV utility scaled by 1500;
- SPNNIV1 frozen-control representation;
- mature legacy action vocabulary;
- external-sampling Deep CFR with repaired all-nonpositive regret fallback;
- separate 3H and HU brains;
- sampled AveragePolicy trajectories;
- 2,000,000-sample reservoir capacity per memory per domain;
- 600 roots per iteration;
- 100 Advantage optimizer steps per domain per iteration;
- batch size 1024;
- 4000 AveragePolicy optimizer steps per domain at milestone finalization.

Admitted Ryzen execution profile:

- 31 root workers;
- one numerical-library thread per root worker;
- 8 parent Torch threads;
- vectorized batch construction;
- production `concurrent_fit` iteration mode.

## Preserved milestones

### LT1

1.2M roots. Preserve:

`/home/rz9/spincore_lean_functional/runs/long_training_lt1/20260915_181249/checkpoint.pt`

SHA256 `beef9bee9439de8d9153450d190e62b0388a678b0778c4185108361f626b4337`.

### LT2 Stage A

1.8M roots / iteration 3000. Preserve:

`/home/rz9/spincore_lean_functional/runs/long_training_lt2/20260916_015559/checkpoint.pt`

SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

At this point 3H AveragePolicy had crossed 2M, HU was only 820,667, and weak-baseline learning was positive overall/3H while HU was flat/noisy.

### LT2 Stage B

4.5M roots / iteration 7500. Preserve:

`/home/rz9/spincore_lean_functional/runs/long_training_lt2_stage_b/20260917_004911/checkpoint.pt`

SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Final sample state:

- 3H roots 2,452,500;
- HU roots 2,047,500;
- 3H Advantage seen 76,144,669;
- HU Advantage seen 62,622,782;
- 3H AveragePolicy seen 5,549,800;
- HU AveragePolicy seen 2,072,704.

Thus all four 2M memories are in replacement regime.

Resource gate:

- final checkpoint 2.623474 GiB;
- finalized save 72.938 s;
- min WSL MemAvailable 7.473 GiB;
- max swap 0 GiB;
- source checkpoint preserved.

## Stage A -> Stage B paired learning review

The two finalized policies were evaluated on the same 1000 fixed-seed empirical scenarios/deals against uniform-legal, passive-caller and jammer families. Checkpoint deltas were paired on scenario, deal, hero seat, opponent family and hero random stream, with 95% CI clustered by scenario.

All nine Stage-B-minus-Stage-A confidence intervals crossed zero:

- uniform legal — ALL +0.425 `[-3.592,+4.442]`; 3H -1.020 `[-6.285,+4.245]`; HU +2.087 `[-4.074,+8.248]`;
- passive caller — ALL -0.795 `[-4.506,+2.917]`; 3H -2.115 `[-6.778,+2.548]`; HU +0.725 `[-5.187,+6.636]`;
- jammer — ALL +1.073 `[-2.428,+4.574]`; 3H +2.974 `[-2.431,+8.380]`; HU -1.115 `[-5.356,+3.126]`.

The extra 2.7M roots therefore did not yield a statistically distinguishable weak-baseline improvement or regression.

The previous HU caveat is now resolved: HU AveragePolicy is saturated, yet no detectable gain appeared.

This is not evidence of exploitability/GTO convergence. Weak fixed opponents may be too insensitive to reveal strategically meaningful policy movement.

## Training decision

Pause same-regime extension beyond iteration 7500.

Do not spend another multi-million-root block merely because resources are healthy. Before more roots or architecture changes, determine whether the AveragePolicy actually moved materially from Stage A to Stage B.

## Immediate read-only policy-drift gate

Run:

`tools/run_lt2_policy_drift_review.sh`

Method:

- export compact finalized Stage A and Stage B policies;
- sample fixed empirical scenarios;
- generate solver states using a checkpoint-independent uniform-legal probe trajectory;
- query both policies on every identical nonterminal state;
- report total-variation distance, argmax disagreement, entropy and max-probability movement;
- break down by 3H/HU and street.

Interpretation:

- tiny drift plus flat EV: current training dynamics are practically stagnant/converged at this representation/optimization regime; investigate training architecture/dynamics before more roots;
- material drift plus flat EV: weak-baseline sentinel has become insensitive; prioritize faithful DeepCrusher or richer cross-play before changing the trainer;
- concentrated/intermediate drift: identify the domain/street where movement occurs before deciding.

This drift gate is diagnostic only, not a strength metric.

## Next possible branches after drift review

If training dynamics look stagnant, inspect Advantage reset/fitting dynamics, reservoir age/composition, policy-target movement, network capacity/representation and optimizer budget before authorizing additional roots.

If policy movement is substantial, keep the current checkpoint and shift effort to stronger evaluation: faithful DeepCrusher R8 v22 direct paired chip-EV, HU/3H separately, then blind/stack/position breakdowns and later full-tournament performance.

Do not enlarge or shrink the 2M reservoirs on intuition alone.

## Operational files

- `tools/run_long_training_lt2_stage_b.sh` — completed Stage B launcher;
- `tools/audit_completed_lt2_stage_b.sh` — Stage B postvalidation PASS;
- `tools/run_lt2_stage_b_learning_review.sh` — completed paired weak-baseline review;
- `tools/compare_lean_checkpoint_delta.py` — direct paired checkpoint delta;
- `tools/run_lt2_policy_drift_review.sh` — immediate next read-only gate;
- `tools/compare_lt2_policy_drift.py` — decision-level policy movement diagnostic;
- `tools/run_lean_functional_training.py` — authoritative trainer.

## Immediate direction

1. Preserve Stage A and Stage B checkpoints.
2. Do not continue training beyond iteration 7500 yet.
3. Pull current `main`.
4. Run `bash tools/run_lt2_policy_drift_review.sh`.
5. Review the policy-drift JSON together with the flat paired weak-baseline delta.
6. Then choose between training-dynamics investigation and stronger benchmarking.
7. Continue faithful DeepCrusher oracle construction in parallel.
