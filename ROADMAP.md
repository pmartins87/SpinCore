# SpinCore Roadmap — active state 2026-09-17

This file tracks the active legacy-first functional training path. Historical snapshots remain preserved in Git history and validation/docs; they do not override the current plan.

## Active status

- LT0 calibration — **DONE**: 120k roots.
- LT1 production-shaped milestone — **DONE**: 1.2M roots.
- LT1 physical fit optimization — **PASS**: 31 root workers, 8 parent Torch threads, vectorized batching.
- LT2 Stage A — **PASS**: iteration 3000 / 1.8M roots.
- Concurrent-fit production parity — **PASS**: exact semantic parity.
- LT2 Stage B — **PASS**: iteration 7500 / 4.5M roots.
- LT2 Stage B resource/postvalidation — **PASS**: zero swap; finalized checkpoint valid; all four 2M reservoirs saturated/replacement.
- Stage A -> Stage B paired weak-baseline review — **COMPLETE: NO DETECTABLE IMPROVEMENT OR REGRESSION**.
- Decision-level Stage A -> Stage B policy-drift review — **NEXT**.
- DeepCrusher faithful oracle — **BUILD IN PARALLEL**.

Canonical current files:

- `CURRENT_WORK.md`
- `docs/LONG_TRAINING_PLAN.md`
- `docs/LT2_STAGE_B_LEARNING_REVIEW_20260917.md`
- `docs/LT2_STAGE_B_RESOURCE_REVIEW_20260917.md`

## Completed learning milestones

### LT0

120k roots proved the repaired pipeline trains, saves, resumes and plays complete 3H/HU hands.

### LT1

1.2M roots established the production-shaped large-reservoir line.

### LT2 Stage A

1.8M roots. 3H AveragePolicy had crossed 2M while HU remained only 820,667. Weak-baseline trend was positive overall/3H; HU was noisy/flat.

### LT2 Stage B

4.5M roots. Final checkpoint:

`/home/rz9/spincore_lean_functional/runs/long_training_lt2_stage_b/20260917_004911/checkpoint.pt`

SHA256:

`3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`

Final policy sample counts:

- 3H: 5,549,800;
- HU: 2,072,704.

All four 2M memories are now in replacement regime. Resource gate passed with 0 swap and min observed WSL MemAvailable 7.473 GiB.

## Stage A -> Stage B paired learning result

The exact Stage A and Stage B finalized policies were evaluated on the same 1000 fixed-seed empirical scenarios/deals and weak opponent families, with a direct paired checkpoint-delta CI clustered by scenario.

Stage B - Stage A deltas (chips/hand):

- uniform legal: ALL +0.425 `[-3.592,+4.442]`; 3H -1.020 `[-6.285,+4.245]`; HU +2.087 `[-4.074,+8.248]`;
- passive caller: ALL -0.795 `[-4.506,+2.917]`; 3H -2.115 `[-6.778,+2.548]`; HU +0.725 `[-5.187,+6.636]`;
- jammer: ALL +1.073 `[-2.428,+4.574]`; 3H +2.974 `[-2.431,+8.380]`; HU -1.115 `[-5.356,+3.126]`.

All nine 95% CIs include zero. The extra 2.7M roots therefore produced no statistically distinguishable gain or regression under this weak-baseline sentinel.

This closes the previous excuse that HU had not yet saturated: HU now crossed 2M and still showed no measurable weak-baseline improvement.

Do not infer GTO convergence from this. Weak fixed opponents may be insensitive to policy movement.

## Immediate decision gate

Do **not** auto-extend beyond iteration 7500.

Run `tools/run_lt2_policy_drift_review.sh`.

The read-only diagnostic compares Stage A and Stage B action distributions on identical checkpoint-independent uniform-legal probe trajectories and reports total-variation distance, argmax disagreement and entropy/max-probability movement overall, by domain and by street.

Decision logic:

- tiny drift -> investigate training dynamics/architecture before more roots;
- material drift -> weak-baseline evaluator likely lacks sensitivity; prioritize faithful DeepCrusher / richer cross-play before altering training;
- intermediate drift -> inspect domain/street concentration first.

## Product strength path

Future product evidence must include faithful DeepCrusher R8 v22 paired chip-EV, HU/3H separately, and later stack/blind/position and full-tournament breakdowns. Weak baselines remain regression sentinels only.

## Immediate action

```bash
bash tools/run_lt2_policy_drift_review.sh
```

Wait for `LT2_POLICY_DRIFT_REVIEW_PASS`, send the generated `SpinCore_LT2A_to_LT2B_policy_drift.json`, and do not start further training first.
