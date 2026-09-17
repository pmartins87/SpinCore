# SpinCore Current Work

Date: 2026-09-17
Status: **LT2 STAGE B PASS — 4.5M ROOTS — HU-JAMMER NEGATIVE — SAME-INPUT DUPLICATES TOO SPARSE — HU PREFLOP CONDITIONAL RESAMPLING NEXT**

## Active source of truth

Read before new compute:

- `docs/LT2_SAME_INPUT_TARGET_VARIANCE_RESULT_20260917.md`
- `docs/LT2_HU_PREFLOP_CONDITIONAL_RESAMPLING_20260917.md`
- `docs/LT2_REPEATED_TARGET_VARIANCE_RESULT_20260917.md`
- `docs/LT2_ADVANTAGE_VALUE_SENSITIVITY_RESULT_20260917.md`
- `docs/LT2_WEAK_BASELINE_VARIANCE_RESULT_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`

Preserve Stage A and Stage B. Do not continue root training beyond iteration 7500 until the targeted HU-preflop conditional decomposition is reviewed.

## Preserved checkpoints

Stage A: iteration 3000 / 1.8M roots, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Confirmed practical failure

The powered 30k weak-baseline gate established:

- Stage B HU Jammer raw chip EV `-5.141`, simultaneous family-wise 95% CI `[-9.078,-1.204]`;
- Stage-B-minus-Stage-A HU-Jammer paired delta `-1.682`, simultaneous six-claim interval approximately `[-3.143,-0.222]`.

Root scaling remains paused.

## Closed optimizer findings

AveragePolicy: extra fitting does not improve held-out CE.

Advantage: 100 -> 1600 steps reduces held-out MSE by only about 3.5–3.8% and does not reproducibly improve production-policy TV. A global optimizer-budget increase is not admitted.

## Repeated-state variance result

For fixed exact state/deal, production `exact_opponent_levels=0` attributes about 35.1% of aggregate sampled-target MSE in 3H and 36.7% in HU to opponent-action sampling noise.

`exact_opponent_levels=1` reduces that measured variance by about 78.0% in 3H and 80.5% in HU at node-cost multipliers 2.268x and 2.306x. For that isolated noise source, level 1 is about 2.00x and 2.22x more statistically efficient per node.

The remaining fixed-deal residual is not pure network error because hidden opponent cards and future board are not part of the information-set input.

## Same-input reservoir result

The complete Stage-B Advantage reservoirs were grouped by exact SPNNIV1 observation bytes plus exact legal mask.

Exact duplicates are too sparse for representative inference.

Threshold >=2 item coverage:

- 3H preflop 0.0368%, flop 0.0214%, turn 0.0716%, river 0.1150%;
- HU preflop 0.2348%, flop 0.0195%, turn 0.0506%, river 0.1464%.

No group reached size 8. HU preflop max group size is 3. Only one 3H preflop group reached size 4.

Postflop duplicate groups have zero maximum iteration span in this run, so they mostly reflect within-iteration duplication rather than independent historical revisits.

Among the sparse cross-iteration preflop duplicates, the decomposition is suggestive but not globally admissible:

- 3H preflop: within-input variance 33.8%, model-to-mean 66.2%;
- HU preflop: within-input variance 36.9%, model-to-mean 63.1%.

HU preflop target/model action mass on that sparse subset remains directionally concerning:

- FOLD 38.2% target mean vs 16.6% model;
- CHECK_CALL 23.7% vs 44.8%;
- ALL_IN 23.2% vs 36.4%.

Do not extrapolate those percentages to the full preflop distribution because coverage is only 0.2348%.

## Immediate bounded gate — HU preflop conditional resampling

Run:

```bash
bash tools/run_lt2_hu_preflop_conditional_resampling.sh
```

Design:

- HU preflop only, because the confirmed strength failure is HU Jammer and exact-duplicate evidence is best there;
- 64 observable anchor states: 16 root, 32 continuation-1, 16 continuation-2+;
- preserve exact hero cards, public state, action history and SPNNIV1 observation;
- enumerate all 2450 ordered opponent hands and weight them by the current Stage-B opponent behavior likelihood of the observed public path;
- sample 16 opponent hands by posterior stratification;
- sample 4 future boards per hand;
- recompute 4 targets per exact hidden deal with `exact_opponent_levels=1`;
- no training roots, no optimizer steps, source checkpoint read only.

Balanced finite-sample decomposition:

`sample-target MSE`
`= within-deal opponent-action variance`
`+ future-board variance`
`+ opponent-hand posterior variance`
`+ model MSE to the current conditional mean`.

This directly separates target variability that one deterministic information-set prediction cannot fit from current-model conditional-mean error.

## Decision after conditional resampling

- If hidden-hand/future-board variance dominates, prioritize target estimators/variance reduction and information-set-correct learning design, not a larger network.
- If model error to the conditional mean dominates, representation/capacity/per-action calibration becomes a strong causal candidate. Localize HU FOLD vs CHECK_CALL/ALL_IN, especially states facing ALL_IN.
- If within-deal action noise remains large despite exact level 1, revisit branching depth.
- Any candidate must later beat preserved Stage B on the powered weak-baseline suite before long root scaling resumes.

DeepCrusher remains deferred.

## Immediate user action

Pull current `main` and run `bash tools/run_lt2_hu_preflop_conditional_resampling.sh`. Wait for `LT2_HU_PREFLOP_CONDITIONAL_RESAMPLING_PASS`, then send `SpinCore_LT2_hu_preflop_conditional_resampling.json`. Do not resume root training first.
