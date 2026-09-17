# SpinCore Roadmap — active state 2026-09-17

## Active status

- LT0 calibration — **DONE**: 120k roots.
- LT1 production-shaped milestone — **DONE**: 1.2M roots.
- LT1 physical fit optimization — **PASS**.
- LT2 Stage A — **PASS**: 1.8M roots.
- LT2 Stage B — **PASS**: 4.5M roots / iteration 7500.
- Stage B resource gate — **PASS**.
- Policy drift Stage A -> Stage B — **MATERIAL MOVEMENT CONFIRMED**.
- Checkpoint cross-play — **NO REPRODUCIBLE ORDERING**.
- 30k weak-baseline gate — **COMPLETE; PRECISION TARGET MET; HU JAMMER NEGATIVE**.
- Stored-target fit audit — **COMPLETE**.
- First held-out fit-budget sweep — **COMPLETE**.
- AveragePolicy extra-budget hypothesis — **NOT SUPPORTED BY HELD-OUT CE CURVE**.
- Advantage 100-step sufficiency — **NOT STRICTLY PLATEAUED IN MSE; BEHAVIOR EFFECT UNRESOLVED**.
- First Advantage policy-TV/argmax diagnostic — **INVALID FOR PRODUCTION SEMANTICS DUE HISTORICAL UNIFORM FALLBACK IN AUDIT HELPER**.
- Corrected production-semantics multi-seed Advantage budget sweep V2 — **NEXT**.
- Root training beyond iteration 7500 — **PAUSED**.
- DeepCrusher — **DEFERRED**.

Canonical current files:

- `CURRENT_WORK.md`
- `docs/LT2_FIT_BUDGET_SWEEP_RESULT_20260917.md`
- `docs/LT2_TRAINING_DYNAMICS_FIT_RESULT_20260917.md`
- `docs/LT2_WEAK_BASELINE_VARIANCE_RESULT_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`

## Preserved checkpoints

Stage A: iteration 3000 / 1.8M roots, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Keep both unchanged.

## Why roots remain paused

The powered weak-baseline evaluation established:

- Stage B HU Jammer `-5.141`, simultaneous family-wise 95% CI `[-9.078,-1.204]`;
- Stage B minus Stage A HU Jammer `-1.682`, simultaneous six-claim CI approximately `[-3.143,-0.222]`.

The policy changed materially but this weak-opponent HU cell became worse. Root count alone is therefore not a justified next intervention.

## What the first budget sweep established

### AveragePolicy

Additional optimizer budget did not improve held-out CE in either domain.

3H: stored `1.089719`; +4000 `1.090321`; +8000 `1.092875`.

HU: stored `1.116161`; +4000 `1.117772`; +8000 `1.119875`.

This rejects the simple hypothesis that the current problem is mainly insufficient AveragePolicy optimizer steps. It does not reject architecture/representation/target limitations.

### Advantage MSE

The MSE curve continues to improve after the canonical 100 steps, but slowly.

3H: `0.031933` at 100 -> `0.031279` at 1600, about 2.0% relative reduction.

HU: `0.045925` at 100 -> `0.044057` at 1600, about 4.1% relative reduction.

So 100 is not a strict MSE plateau, especially in HU. Whether that extra MSE reduction translates into better behavior remains unresolved.

## Diagnostic correction

The first fit-audit helper converted Advantage vectors to policies with the old `regret_matching_policy()` semantics: uniform legal fallback when all legal advantages were non-positive.

Functional production SpinCore instead installs `LeanNeuralActionAdvantagePolicy`, whose fallback is a stable masked softmax over raw legal advantages. The first release training specification requires this repaired behavior.

Therefore:

- Advantage MSE metrics from the first audit/sweep are valid;
- AveragePolicy metrics are valid;
- Advantage-derived TV/argmax metrics from those reports are not canonical production-policy metrics and must not drive decisions.

This mismatch was in the diagnostic only; current production training already uses the repaired Lean adapter.

## Immediate gate — V2 Advantage budget sweep

Launcher: `tools/run_lt2_stage_b_advantage_budget_sweep_v2.sh`.

Design:

- Stage B checkpoint read only;
- no roots;
- fixed 25k holdout per domain, excluded from optimization;
- cumulative budgets `0,25,50,100,200,400,800,1600`;
- production Lean positive-regret + all-nonpositive masked-softmax semantics;
- three independent deterministic reset/training replicates;
- per-replicate and aggregate MSE, zero-gap fraction, production-policy TV, argmax and all-nonpositive frequency.

This is intentionally multi-seed so a budget decision is not made from one network initialization.

## Branch after V2

If production-policy TV/argmax improves reproducibly beyond 100 along with MSE, test a larger Advantage budget in a small isolated continuation; compare it against preserved Stage B using the existing powered weak-baseline design before scaling roots.

If MSE improves but production-policy metrics do not, do not spend more optimizer budget blindly. Move to target-noise/conditional-variance, SPNNIV1 aliasing/capacity and regret-objective sensitivity diagnostics.

If the corrected curve plateaus early on both loss and behavior metrics, optimizer budget is not the main bottleneck; investigate representation and target generation directly.

## Immediate action

```bash
bash tools/run_lt2_stage_b_advantage_budget_sweep_v2.sh
```

Do not resume root training until the V2 curve is reviewed.
