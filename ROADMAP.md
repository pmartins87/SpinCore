# SpinCore Roadmap — active state 2026-09-19

## Active status

- LT0 — **DONE**.
- LT1 — **DONE**.
- LT2 Stage A — **PASS**.
- LT2 Stage B — **PASS** at 4.5M roots / iteration 7500.
- Weak-baseline regression — **CONFIRMED**.
- AveragePolicy extra-fit — **NOT SUPPORTED**.
- Advantage optimizer escalation — **NOT SUPPORTED**.
- HU-preflop target variance — **HIDDEN/CHANCE DOMINANT**.
- K4 board averaging — **VALID ESTIMATOR IMPROVEMENT**.
- Jammer K4 causal gate — **NOT MET**.
- Cross-street future-chance audit — **PASS; FUTURE-CHANCE NOT FAILURE-SPECIFIC**.
- Stage-A/B target-drift matrix — **PASS; NO UNIVERSAL TARGET-DRIFT EXPLANATION**.
- HU policy-chain audit — **PASS; JAMMER DEFECT UPSTREAM IN CURRENT BEHAVIOR**.
- HU current-behavior first divergence — **PASS; 73.86% OF JAMMER LOSS AT PREFLOP FACING ALL-IN**.
- Broad Jammer FAI action-gap/RM calibration — **NEXT**.
- K4 training — **NOT AUTHORIZED**.
- long root training — **PAUSED**.
- DeepCrusher — **DEFERRED**.

## Preserved checkpoints

Stage A SHA:
`e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B SHA:
`3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Current-behavior localization verdict

Jammer BEH B-A:
- total `-8.4304`, CI95 `[-12.5155,-4.3453]`.

First-divergence contribution:
- FAI `-6.2267`, CI95 `[-9.2575,-3.1959]`;
- ROOT `-2.2037`, CI crosses zero;
- all postflop groups exactly zero.

FAI explains **73.86%** of the resolved current-behavior loss.

## Why MSE is no longer enough

Prior Jammer FAI diagnostics showed:
- common target stationary A->B;
- no matching aggregate own-target MSE degradation;
- yet current behavior loses heavily.

Production regret matching has a nonlinear zero boundary:
- positive regrets define support;
- all-nonpositive vectors trigger softmax fallback.

Therefore small sign/ranking errors can have large policy effects while MSE remains similar.

## Next gate

Run `tools/run_lt2_jammer_fai_broad_calibration.sh`.

Selection:
- common Jammer FAI states reached before prior A/B divergence;
- sample before the FAI hero action;
- no conditioning on FAI divergence or outcome;
- 8 anchors/seed, 48 total.

Measure:
- common low-noise Q action gaps;
- stage-specific true regrets;
- raw Advantage MSE;
- canonical gap MSE;
- positive-support agreement;
- false positive / false negative support;
- fallback incidence;
- best-action agreement;
- mass on truly negative actions;
- policy regret.

## Decision after calibration

If Stage B has similar MSE but worse support/sign calibration or higher policy regret:
- investigate RM-sensitive training objectives / calibration.

If Stage B is not worse on broad FAI states:
- investigate trajectory weighting and root interaction instead.

No training before this gate.

Holdout `20261001..20261006` remains sealed.

## Immediate action

Run `bash tools/run_lt2_jammer_fai_broad_calibration.sh`.

Stop at PASS or first error. Do not train.
