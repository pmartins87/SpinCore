# SpinCore Current Work

Date: 2026-09-18
Status: **LT2 STAGE B PASS — HU-JAMMER REGRESSION LOCALIZED — K4 MECHANICS PASS — V1 OVERLAY DIRECTIONALLY POSITIVE BUT REFERENCE-MISMATCHED — COMMON-REFERENCE V2 NEXT — NO TRAINING**

## Active source of truth

Read before new compute:

- `docs/LT2_STAGE_A_B_FIRST_DIVERGENCE_RESULT_20260918.md`
- `docs/LT2_JAMMER_FACING_ALLIN_TARGET_OVERLAY_20260918.md`
- `docs/LT2_JAMMER_FACING_ALLIN_OVERLAY_V1_CORRECTION_20260918.md`
- `docs/LT2_HU_PREFLOP_BOARD_AVERAGING_SMOKE_RESULT_20260918.md`
- `docs/LT2_HU_PREFLOP_BOARD_ONLY_AVERAGING_RESULT_20260918.md`
- `docs/LT2_HU_PREFLOP_TARGET_ESTIMATOR_BUDGET_RESULT_20260917.md`
- `docs/LT2_WEAK_BASELINE_VARIANCE_RESULT_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`

Preserve Stage A and Stage B. Do not continue root training beyond iteration 7500.

## Preserved checkpoints

Stage A: iteration 3000 / 1.8M roots, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Confirmed Stage-B strength regression

Powered weak-baseline gate:
- HU Jammer Stage B raw EV `-5.141`, simultaneous 95% CI `[-9.078,-1.204]`;
- Stage-B-minus-Stage-A HU Jammer `-1.682`, CI `[-2.767,-0.597]`;
- PassiveCaller HU Stage-B-minus-Stage-A `-1.261`, CI `[-2.377,-0.145]`.

Long root scaling remains paused.

## Target-estimator evidence

HU-preflop target decomposition:
- future-board variance **65.88%**;
- opponent-hand variance **26.10%**;
- exact-level-1 residual action noise **1.74%**;
- model-to-conditional-mean error **6.27%**.

Exact1 is not compute-efficient. Board-only K4 is the measured estimator compute elbow and its implementation has passed a strict mechanics smoke.

This still does not by itself prove K4 caused or fixes the deployed-policy regression.

## Stage-A -> Stage-B first-divergence forensic — COMPLETE

Read-only forensic:
- seeds `20260920..20260925`;
- 13,585 HU scenarios;
- 81,510 seat-runs;
- UNIFORM_LEGAL, PASSIVE_CALLER, JAMMER;
- paired scenario/deal/hero-seat/random streams.

### JAMMER

Overall:
- B-A `-1.682`;
- 95% CI `[-2.767,-0.597]`;
- first-divergence rate only `4.8%`.

Contribution by first divergence:
- PREFLOP_ROOT: frequency `2.3%`, contribution `-0.557`, CI `[-1.139,+0.026]`;
- PREFLOP_FACING_ALL_IN: frequency `2.5%`, contribution `-1.126`, CI `[-2.049,-0.202]`.

Thus the resolved FACING_ALL_IN component alone explains about **67%** of the total Jammer regression.

There are no postflop first-divergence contributions in the Jammer result.

### PASSIVE_CALLER

Overall:
- B-A `-1.261`, CI `[-2.377,-0.145]`.

Largest resolved component:
- FLOP contribution `-0.672`, CI `[-1.299,-0.046]`.

So the Stage-B regression is **not a universal HU-preflop phenomenon**.

### UNIFORM_LEGAL

Overall B-A is unresolved:
- `-0.416`, CI `[-1.812,+0.979]`.

A resolved negative TURN subgroup exists:
- contribution `-0.552`, CI `[-0.970,-0.134]`.

## Current interpretation

The Jammer regression independently localizes to the same broad state class where lower-variance target diagnostics had already found a problem: HU preflop after opponent ALL_IN.

This substantially strengthens K4 as a **Jammer-facing causal candidate**.

However:
- K4 is not established as a global fix;
- PassiveCaller exposes a separate postflop regression;
- the causal chain from noisy Advantage target -> current Advantage -> accumulated AveragePolicy is still incomplete.

## V1 target overlay — completed, but not final causal evidence

V1 on 24 actual Jammer FACING_ALL_IN first-divergence states reported:
- Stage A avgTV `0.3647`, advTV `0.5101`, K1TV `0.3283`, K4TV `0.2706`;
- Stage B avgTV `0.4111`, advTV `0.5963`, K1TV `0.3524`, K4TV `0.2592`;
- B-A avgTV error `+0.0464`;
- B-A advTV error `+0.0862`;
- B-A Advantage regret `+10.70` chips.

This is directionally supportive, but review found a conceptual mismatch: V1 compared A and B against **different stage-specific self-play posteriors**.

The actual JAMMER policy is hand-independent, so conditioning on the observed jam must not reweight opponent private cards. Stage A and Stage B therefore need one **common Jammer-conditioned reference** for a direct benchmark comparison.

V1 is retained as descriptive training-process evidence only.

## Corrected final causal gate — COMMON REFERENCE V2

Run:

```bash
bash tools/run_lt2_jammer_facing_allin_common_reference_v2.sh
```

V2:
- reconstructs the same class of actual forensic Jammer FACING_ALL_IN states;
- uses uniform compatible opponent hands because JAMMER is hand-independent;
- uses uniform future boards;
- builds one common target reference shared by Stage A and Stage B;
- asserts that fixed-deal targets computed with Stage A and Stage B runtimes are identical after opponent is already all-in;
- compares A/B AveragePolicy and Advantage to that same target;
- compares K1 vs K4 estimator error against the same target.

Reserved holdout seeds `20261001..20261006` remain untouched.

## Decision after V2

Only if the same common-reference comparison shows:
1. Stage B AveragePolicy farther from target than Stage A;
2. Stage B Advantage policy farther from target in a coherent direction;
3. action-mass shifts explain the Jammer-facing regression;
4. K4 improves estimator quality on those exact states;

may a bounded K4 pilot be considered.

Even then, long training remains paused until the separate postflop regression is accounted for.

DeepCrusher remains deferred.

## Immediate user action

Pull current `main` and run `bash tools/run_lt2_jammer_facing_allin_common_reference_v2.sh`.

Wait for `LT2_JAMMER_FACING_ALLIN_COMMON_REFERENCE_V2_PASS` or the first error.

Then send `SpinCore_LT2_jammer_facing_allin_common_reference_v2.json`.

Do not start K4 training.
