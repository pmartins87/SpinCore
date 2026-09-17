# SpinCore Current Work

Date: 2026-09-17
Status: **LT2 STAGE B PASS — 4.5M ROOTS — HU-JAMMER NEGATIVE CONFIRMED — POLICY FIT BUDGET PLATEAUS — ADVANTAGE MSE IMPROVES MODESTLY — CORRECTED MULTI-SEED ADVANTAGE SWEEP NEXT**

## Active source of truth

Read before new compute:

- `docs/LT2_FIT_BUDGET_SWEEP_RESULT_20260917.md`
- `docs/LT2_TRAINING_DYNAMICS_FIT_RESULT_20260917.md`
- `docs/LT2_WEAK_BASELINE_VARIANCE_RESULT_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`

Preserve Stage A and Stage B. Do not continue root training beyond iteration 7500 until the corrected Advantage budget curve is reviewed.

## Preserved checkpoints

Stage A: iteration 3000 / 1.8M roots, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Established strength result

The powered 30k weak-baseline gate established a real current failure:

- Stage B HU Jammer raw chip EV `-5.141`, simultaneous family-wise 95% CI `[-9.078,-1.204]`;
- Stage-B-minus-Stage-A HU-Jammer paired delta `-1.682`, simultaneous six-claim interval approximately `[-3.143,-0.222]`.

Thus extra Stage-A -> Stage-B roots measurably worsened this cell. More roots are not admitted blindly.

## Held-out fit-budget sweep result

The first Stage-B held-out sweep completed with source checkpoint unchanged.

### AveragePolicy

Additional fit does not improve held-out cross-entropy.

3H CE:

- +0 `1.089719`;
- +1000 `1.090000`;
- +2000 `1.090876`;
- +4000 `1.090321`;
- +8000 `1.092875`.

HU CE:

- +0 `1.116161`;
- +1000 `1.116908`;
- +2000 `1.118689`;
- +4000 `1.117772`;
- +8000 `1.119875`.

Conclusion: insufficient AveragePolicy optimizer budget is not supported as the dominant current mechanism. A plateau with large residual error still leaves target variation, representation aliasing, capacity and objective issues open.

### Advantage MSE

The MSE metrics are valid and show continued but modest improvement after the production 100-step point.

3H:

- 100 steps: MSE `0.031933`, fit-vs-zero `10.62%`;
- 1600: MSE `0.031279`, fit-vs-zero `12.45%`;
- ~`2.0%` relative MSE reduction from 100 -> 1600.

HU:

- 100 steps: MSE `0.045925`, fit-vs-zero `14.01%`;
- 1600: MSE `0.044057`, fit-vs-zero `17.51%`;
- ~`4.1%` relative MSE reduction from 100 -> 1600.

Therefore 100 steps are not a strict MSE plateau, especially in HU, but this does not yet prove that a larger budget improves behavior or poker strength.

## Important diagnostic correction

The first fit audit/sweep computed Advantage-derived TV/argmax with the historical **uniform** fallback when all legal predicted advantages were non-positive.

Production functional SpinCore uses `LeanNeuralActionAdvantagePolicy`: positive-regret matching, but **masked softmax over raw legal advantages** when none are positive.

This is a **diagnostic metric bug only**, not a production-training fallback bug. The trainer installs the corrected Lean behavior policy.

Consequences:

- previous Advantage MSE/zero-baseline results remain valid;
- previous AveragePolicy metrics remain valid;
- previous Advantage-derived TV/argmax values must not be used to decide the fit budget.

The audit helper is now corrected and schema-bumped to V2.

## Immediate bounded gate

Run:

```bash
bash tools/run_lt2_stage_b_advantage_budget_sweep_v2.sh
```

Design:

- Stage B only;
- no roots and no source checkpoint mutation;
- deterministic 25k held-out Advantage sample per domain;
- holdout excluded from optimization;
- budgets `0,25,50,100,200,400,800,1600`;
- three independent deterministic reset/training replicates;
- exact production Lean regret-matching + softmax-fallback semantics for policy-space metrics;
- aggregate mean/range/stdev across replicates.

## Decision after V2

If corrected production-semantics policy error improves reproducibly beyond 100 steps, test a larger Advantage budget in a **small isolated continuation**, benchmarked against preserved Stage B before any long run.

If MSE improves but production-policy TV/argmax does not, extra optimizer steps are not addressing the behavior-level problem; move to target-variance / representation / objective diagnostics.

If both MSE and policy metrics plateau, move directly to target generation, SPNNIV1 aliasing/capacity and HU-specific state/action concentration.

DeepCrusher remains later and is not a current dependency.

## Immediate user action

Pull current `main` and run `bash tools/run_lt2_stage_b_advantage_budget_sweep_v2.sh`. Wait for `LT2_STAGE_B_ADVANTAGE_BUDGET_SWEEP_V2_PASS`, then send `SpinCore_LT2_stage_b_advantage_budget_sweep_v2.json`. Do not resume root training first.
