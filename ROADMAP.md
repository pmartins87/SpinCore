# SpinCore Roadmap — active state 2026-09-17

## Active status

- LT0 calibration — **DONE**: 120k roots.
- LT1 production-shaped milestone — **DONE**: 1.2M roots.
- LT1 physical fit optimization — **PASS**: 31 root workers, 8 parent Torch threads, vectorized batching.
- LT2 Stage A — **PASS**: 1.8M roots.
- Concurrent-fit production parity — **PASS**.
- LT2 Stage B — **PASS**: 4.5M roots / iteration 7500.
- Stage B resource gate — **PASS**.
- Policy drift Stage A -> Stage B — **MATERIAL MOVEMENT CONFIRMED**.
- Checkpoint cross-play — **NO REPRODUCIBLE ORDERING**.
- 30k weak-baseline gate — **COMPLETE; PRECISION TARGET MET; HU JAMMER NEGATIVE**.
- Read-only stored-target fit audit — **COMPLETE; LARGE APPROXIMATION BURDEN OBSERVED, BUT BUDGET SUFFICIENCY UNRESOLVED**.
- Held-out Stage-B optimizer-budget sweep — **NEXT**.
- Root training beyond iteration 7500 — **PAUSED**.
- DeepCrusher — **DEFERRED TO LATER ADVANCED BENCHMARK**.

Canonical current files:

- `CURRENT_WORK.md`
- `docs/LT2_TRAINING_DYNAMICS_FIT_RESULT_20260917.md`
- `docs/LT2_TRAINING_DYNAMICS_FIT_AUDIT_20260917.md`
- `docs/LT2_WEAK_BASELINE_VARIANCE_RESULT_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`

## Preserved checkpoints

Stage A: iteration 3000 / 1.8M roots, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Keep both unchanged.

## Why roots remain paused

The powered weak-baseline evaluation established a specific failure mode rather than a variance-only artifact:

- Stage B HU Jammer raw EV `-5.141`, simultaneous family-wise 95% CI `[-9.078,-1.204]`;
- Stage B minus Stage A HU Jammer `-1.682`, simultaneous six-claim CI approximately `[-3.143,-0.222]`.

Training changed the policy but made this cell worse. Therefore another long block cannot be justified by root count alone.

## What the fit audit added

The read-only audit sampled 25k items from each Stage A/B Advantage and AveragePolicy reservoir.

Stage B weighted fit diagnostics:

- 3H Advantage: only `9.98%` of zero-predictor MSE removed; induced-policy TV `0.5969`;
- HU Advantage: `14.17%` removed; TV `0.6058`;
- 3H AveragePolicy: `12.98%` of uniform-to-target CE gap closed; KL `0.6323`, TV `0.4286`;
- HU AveragePolicy: `16.27%` gap closed; KL `0.6387`, TV `0.4318`.

These residuals are large, but they cannot be labeled optimizer underfit directly because external-sampling targets and historical AveragePolicy targets contain irreducible conditional variation.

Mechanics also matter: Advantage resets each iteration and receives only 100 x 1024 sample draws from a 2M reservoir; AveragePolicy has 12,000 cumulative optimizer steps at Stage B, with +4000 per milestone finalization.

## Immediate gate — fixed held-out budget curves

Launcher: `tools/run_lt2_stage_b_fit_budget_sweep.sh`.

Design:

- Stage B only;
- no roots and no checkpoint mutation;
- deterministic 25k held-out set per memory;
- held-out items excluded from optimization sampling;
- both 3H and HU.

Advantage: fresh deterministic reset, cumulative budgets `0,25,50,100,200,400,800,1600`; canonical production point = 100.

AveragePolicy: continue stored Stage-B model+optimizer, cumulative **additional** budgets `0,1000,2000,4000,8000`; canonical finalization increment = +4000.

There is no fixed strength cutoff. The diagnostic question is the shape of the held-out curve.

## Branch after budget sweep

If held-out Advantage fit continues improving strongly after 100 steps, optimizer budget is causally implicated. The next experiment will be a small isolated continuation using a larger Advantage budget, benchmarked against the preserved Stage B before any long run.

If held-out AveragePolicy fit improves strongly with extra fitting, create an isolated policy-refit candidate without collecting new roots and test it against the same weak-baseline suite.

If a curve plateaus early while residual fit remains poor, optimizer budget is not the dominant constraint; inspect capacity, SPNNIV1 representation, target variance/aliasing, reservoir weighting and HU state/action concentration.

If both curves plateau early, move directly to target-generation/representation diagnostics rather than root scaling.

## Immediate action

```bash
bash tools/run_lt2_stage_b_fit_budget_sweep.sh
```

Do not resume root training until the resulting held-out curves are reviewed.