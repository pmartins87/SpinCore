# SpinCore — LT2 Stage-B held-out fit-budget sweep result

Date: 2026-09-17
Status: **AVERAGEPOLICY BUDGET NOT IMPLICATED — ADVANTAGE MSE IMPROVES MODESTLY PAST 100 — FIRST ADVANTAGE POLICY METRIC USED WRONG FALLBACK — CORRECTED MULTI-SEED ADVANTAGE SWEEP NEXT**

## Source

Preserved Stage-B checkpoint: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

The completed sweep used a deterministic 25k holdout per memory, excluded held-out items from optimization, collected no roots, and left the source checkpoint unchanged.

## AveragePolicy result is valid

AveragePolicy held-out metrics do not use regret matching, so the first sweep's policy results are valid.

### THREE_HANDED

- stored Stage B / +0: CE `1.089719`, KL `0.633699`, TV `0.4291`, argmax `0.5040`;
- +1000: CE `1.090000`;
- +2000: CE `1.090876`;
- +4000: CE `1.090321`;
- +8000: CE `1.092875`.

There is no held-out CE improvement from additional policy fitting. The curve is effectively flat/slightly worse.

### TRUE_HEADS_UP

- stored Stage B / +0: CE `1.116161`, KL `0.635417`, TV `0.4303`, argmax `0.4952`;
- +1000: CE `1.116908`;
- +2000: CE `1.118689`;
- +4000: CE `1.117772`;
- +8000: CE `1.119875`.

Again, extra policy optimizer steps do not improve held-out CE. Therefore insufficient AveragePolicy optimizer budget is not supported as the current dominant mechanism.

This does **not** prove the AveragePolicy architecture/representation is adequate. A plateau with large residual error can instead indicate target variation, representational aliasing, or capacity/objective limits.

## Advantage MSE result is valid

The legal-action MSE and zero-baseline metrics are independent of the policy fallback and remain valid.

### THREE_HANDED

At the canonical 100-step budget: MSE `0.031933`, fit-vs-zero `10.62%`.

At 1600 steps: MSE `0.031279`, fit-vs-zero `12.45%`.

Thus 15x additional fit after the canonical point reduces held-out MSE by only about `2.0%` relative and closes about `1.83` additional percentage points of the zero-baseline gap.

### TRUE_HEADS_UP

At 100 steps: MSE `0.045925`, fit-vs-zero `14.01%`.

At 1600 steps: MSE `0.044057`, fit-vs-zero `17.51%`.

Thus additional fitting has a clearer but still bounded HU effect: about `4.1%` relative MSE reduction and `3.50` additional percentage points of zero-baseline error removed.

This means the canonical 100 steps are not at a strict MSE plateau, especially in HU. It does **not** yet establish that increasing the budget improves the actual behavior policy or poker strength.

## Diagnostic metric correction

The first audit/sweep's Advantage-derived `TV` and `argmax` metrics used `_rm_policy()` from `tools/audit_lt2_checkpoint_fit.py`. That helper implemented the historical behavior:

- normalize positive regrets;
- if every legal advantage is non-positive, fall back to **uniform legal**.

Production SpinCore does **not** use that fallback. `python/spincore/lean_action_policy.py` uses the repaired Lean semantics:

- normalize positive legal regrets;
- if none are positive, use a stable **masked softmax over raw legal advantages**.

The production trainer installs `LeanNeuralActionAdvantagePolicy` into the functional runtime, so this is a diagnostic-only mismatch, not evidence that Stage B itself trained with the old fallback.

Because all-nonpositive cases are nontrivial in the stored targets/predictions, the first sweep's Advantage policy-TV/argmax curve must not be used to decide whether a larger Advantage fit budget helps behavior.

The MSE curve and all AveragePolicy metrics remain valid.

## Corrective experiment

Run `tools/run_lt2_stage_b_advantage_budget_sweep_v2.sh`.

Design:

- Stage B only;
- no roots;
- source checkpoint read-only and SHA-checked;
- deterministic 25k held-out Advantage sample per domain;
- holdout excluded from optimization;
- budgets `0,25,50,100,200,400,800,1600`;
- **three independent deterministic reset/training replicates** rather than one;
- Advantage policy metrics use the exact production Lean positive-regret + masked-softmax fallback semantics;
- report includes per-replicate curves and aggregate mean/range/stdev.

This simultaneously fixes the fallback mismatch and tests whether the apparent budget effect depends on one initialization.

## Current decision

Do not resume root training.

AveragePolicy optimizer budget is not the leading suspect. Advantage budget remains a live but unproven hypothesis because MSE continues to improve modestly after 100 steps, especially in HU. The corrected multi-seed production-semantics curve is required before deciding whether to test a larger Advantage budget in an isolated continuation or move deeper into target variance/representation diagnostics.
