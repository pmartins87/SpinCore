# SpinCore — LT2 training-dynamics fit audit

Date: 2026-09-17
Status: **COMPLETE — LARGE STORED-TARGET FIT ERROR OBSERVED; HELD-OUT BUDGET SWEEP NEXT**

## Why this audit exists

The 30k multi-seed weak-baseline gate met its predeclared precision target and resolved a real failure mode: Stage B is negative against HU Jammer (`-5.141` chips/hand, simultaneous family-wise 95% CI `[-9.078,-1.204]`). The paired Stage-B-minus-Stage-A HU-Jammer delta is also negative and remains negative after six-claim family-wise correction.

Therefore another blind root block is not justified. The immediate question is whether the neural approximations are limited by optimizer budget before changing root count, model capacity, representation or target semantics.

## Source checkpoints

- Stage A: iteration 3000 / 1.8M roots, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.
- Stage B: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Both were read only.

## Method

`tools/run_lt2_training_dynamics_fit_audit.sh` sampled 25,000 items from each stored Advantage and AveragePolicy memory, separately for 3H/HU. No optimizer step occurred.

Weighted Stage-B results:

- 3H Advantage: error removed vs zero `9.98%`, policy TV `0.5969`, argmax agreement `33.56%`;
- HU Advantage: error removed vs zero `14.17%`, policy TV `0.6058`, argmax agreement `26.40%`;
- 3H AveragePolicy: uniform-to-target gap closed `12.98%`, KL `0.6323`, TV `0.4286`, argmax `51.46%`;
- HU AveragePolicy: uniform-to-target gap closed `16.27%`, KL `0.6387`, TV `0.4318`, argmax `49.20%`.

Stage A was broadly similar. Stage-B HU AveragePolicy is modestly worse than Stage-A HU on KL/TV/argmax/gap-closed, but the audit does not show a new catastrophic fit regime appearing only at Stage B.

Full result and interpretation are recorded in `docs/LT2_TRAINING_DYNAMICS_FIT_RESULT_20260917.md`.

## Interpretation limit

Large per-sample residual error is not identical to optimizer underfitting. Advantage targets contain external-sampling variance. AveragePolicy memory contains historical behavior-policy targets from many iterations, so irreducible conditional variation can remain even for a well-optimized model.

Therefore this audit **implicates approximation burden but does not prove that more optimizer steps solve it**. That question requires a fixed held-out learning curve.

## Mechanics clarified

The current Advantage network is reset from scratch every iteration and then receives 100 optimizer steps at batch size 1024. Against a 2M reservoir that is 102,400 draws; a simple independent-draw coverage calculation corresponds to about 99.8k expected unique items, roughly 5.0% of the reservoir, seen by that fresh network in the iteration.

AveragePolicy `policy_steps=4000` is a per-finalization increment, not lifetime total. Stage A has 8,000 cumulative policy optimizer steps and Stage B has 12,000. The next test therefore measures whether **additional** fitting of the preserved Stage-B policy improves held-out fit.

## Immediate causal gate

Run `tools/run_lt2_stage_b_fit_budget_sweep.sh`.

The sweep is source-read-only, collects no roots and excludes a deterministic 25k holdout from optimization sampling.

Advantage curve: `0,25,50,100,200,400,800,1600` cumulative steps from one fresh deterministic reset. The production 100-step point is included exactly as the reference budget.

AveragePolicy curve: `0,1000,2000,4000,8000` **additional** steps starting from the stored Stage-B model+optimizer. The production +4000 finalization increment is included as the reference scale.

No arbitrary pass threshold is attached to these budgets. The question is whether held-out MSE/KL/TV/argmax continue improving beyond the production budget and where the curve flattens.

## Stop condition

Do not resume root training beyond iteration 7500 until the held-out fit-budget curves are reviewed. DeepCrusher remains deferred.