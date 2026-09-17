# SpinCore — Long-Training Plan

Status: **LT2 STAGE B PASS — ROOT TRAINING PAUSED AT 4.5M — HU-JAMMER FAILURE CONFIRMED — CORRECTED MULTI-SEED ADVANTAGE BUDGET SWEEP ACTIVE**
Date: 2026-09-17

## Current state

The continuous learning line has reached:

- LT0: 120k roots;
- LT1: 1.2M roots;
- LT2 Stage A: 1.8M roots;
- LT2 Stage B: 4.5M roots / iteration 7500;
- all four 2M reservoirs in replacement regime;
- policy drift Stage A -> Stage B is material;
- 30k weak-baseline precision gate complete;
- Stage B HU Jammer is statistically negative and worsened relative to Stage A;
- stored-target fit audit complete;
- first held-out optimizer-budget sweep complete;
- AveragePolicy extra fitting did not improve held-out CE;
- Advantage MSE still improves modestly after 100 steps, especially HU;
- first Advantage-derived policy metrics were computed with the historical uniform all-nonpositive fallback and therefore are not canonical production-policy metrics;
- immediate next experiment is a corrected multi-seed Advantage-only held-out sweep using the actual Lean softmax fallback.

Read `LT2_FIT_BUDGET_SWEEP_RESULT_20260917.md` first.

## Core training contract

Current functional line:

- empirical SpinGo 3H/HU/blind/stack sampling;
- WTA chip-EV utility scaled by 1500;
- SPNNIV1 frozen-control representation;
- mature legacy action vocabulary;
- external-sampling Deep CFR;
- separate 3H and HU brains;
- sampled AveragePolicy trajectories;
- 2,000,000-sample reservoir capacity per memory per domain;
- 600 roots per iteration;
- Advantage reset every iteration;
- 100 Advantage optimizer steps per domain per iteration;
- batch size 1024;
- 4000 AveragePolicy optimizer steps per milestone finalization;
- production behavior uses positive-regret matching and a masked-softmax fallback if all legal predicted advantages are non-positive;
- 31 root workers, one worker numerical thread, 8 parent Torch threads, vectorized batching, production concurrent-fit mode.

## Preserved milestones

LT1: 1.2M roots, SHA256 `beef9bee9439de8d9153450d190e62b0388a678b0778c4185108361f626b4337`.

LT2 Stage A: 1.8M roots / iteration 3000, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

LT2 Stage B: 4.5M roots / iteration 7500, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Keep all preserved milestones unchanged.

## Strength result that freezes root scaling

The powered 30k gate established:

- Stage B HU Jammer `-5.141` chips/hand, simultaneous family-wise 95% CI `[-9.078,-1.204]`;
- Stage B minus Stage A HU Jammer `-1.682`, simultaneous six-claim interval about `[-3.143,-0.222]`.

Therefore the extra Stage-A -> Stage-B training measurably degraded this specific weak-opponent HU cell. More roots cannot be admitted merely because the policy continues to move.

## What the held-out fit work established

### AveragePolicy optimizer budget

Additional fitting from the stored Stage-B policy does not improve held-out cross-entropy.

3H:

- +0 `1.089719`;
- +4000 `1.090321`;
- +8000 `1.092875`.

HU:

- +0 `1.116161`;
- +4000 `1.117772`;
- +8000 `1.119875`.

Thus simply increasing AveragePolicy optimizer steps is not supported as the current fix. The substantial remaining target residual can instead reflect target nonstationarity/variance, model capacity, representation aliasing or objective limits.

### Advantage optimizer budget

Held-out legal-action MSE continues to improve beyond the production 100-step point.

3H:

- 100 steps `0.031933`;
- 1600 steps `0.031279`;
- about 2.0% relative MSE improvement.

HU:

- 100 steps `0.045925`;
- 1600 steps `0.044057`;
- about 4.1% relative MSE improvement.

Thus 100 is not a strict loss plateau, especially in HU. However, the size of this loss improvement is modest and does not by itself show better action selection.

## Correction to the first Advantage policy-space diagnostic

The first audit helper used the historical universal `regret_matching_policy()` conversion for Advantage TV/argmax. Its all-nonpositive fallback is uniform over legal actions.

The functional production runtime explicitly replaces that historical behavior with `LeanNeuralActionAdvantagePolicy`, which uses stable masked softmax over the raw legal advantages when no legal advantage is positive. This matches the canonical first functional training specification.

Therefore the discrepancy is confined to the diagnostic metric. It does not mean Stage B trained with the wrong fallback.

Valid first-sweep evidence:

- Advantage MSE and zero-predictor comparisons;
- all AveragePolicy metrics.

Invalid for production-policy interpretation:

- first-sweep Advantage-induced TV and argmax curves.

The fit-audit helper is corrected and schema-bumped to V2.

## Immediate bounded experiment

Launcher: `tools/run_lt2_stage_b_advantage_budget_sweep_v2.sh`.

Design:

- Stage B source checkpoint read only and SHA-checked before/after;
- no roots;
- deterministic 25k held-out Advantage items per domain;
- held-out items excluded from optimization;
- budgets `0,25,50,100,200,400,800,1600`;
- three independent deterministic reset/training replicates;
- fixed holdout across replicates within domain;
- exact production Lean regret-matching + masked-softmax fallback for behavior-space TV/argmax;
- per-replicate and aggregate mean/min/max/stdev.

The three-replicate design is deliberate: the original single reset showed non-monotone behavior-space metrics, so one initialization is insufficient for a budget decision.

## Decision branches after V2

If larger Advantage budgets reproducibly reduce both held-out MSE and production-policy TV / improve argmax agreement beyond 100, admit only a **small isolated continuation** from Stage B using a selected larger budget. Preserve Stage B, predeclare the comparison, then evaluate the candidate on the same statistically powered weak-baseline suite before any long training.

If MSE continues to improve but production-policy metrics remain flat or worsen, optimizer budget is not fixing the behavior-level discrepancy. Move to target conditional variance/noise, regret-sign sensitivity, SPNNIV1 aliasing/capacity and HU state/action concentration.

If both loss and behavior curves plateau early, skip optimizer changes and inspect representation/target generation directly.

Only after a causal mechanism produces a measurable improvement may root scaling resume.

## DeepCrusher placement

DeepCrusher remains deferred until the weak-opponent curriculum is strong and stable. A literal C++ translation may help later but is not a current dependency.

## Immediate direction

1. Preserve Stage A and Stage B.
2. Keep root training stopped at iteration 7500.
3. Run `bash tools/run_lt2_stage_b_advantage_budget_sweep_v2.sh`.
4. Review `SpinCore_LT2_stage_b_advantage_budget_sweep_v2.json`.
5. Choose between a small larger-budget continuation and deeper target/representation diagnostics from the corrected multi-seed behavior curve.
6. Do not scale roots or move to DeepCrusher yet.
