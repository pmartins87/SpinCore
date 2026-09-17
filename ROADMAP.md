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
- Stage A -> Stage B policy-drift review — **COMPLETE: MATERIAL POLICY MOVEMENT**.
- Initial Stage A -> Stage B contemporary cross-play (3000 scenarios) — **COMPLETE: BORDERLINE / MIXED; NO STAGE-B ADVANTAGE DEMONSTRATED**.
- Higher-power fresh-seed contemporary cross-play (9000 scenarios) — **NEXT**.
- DeepCrusher faithful oracle — **BUILD IN PARALLEL**.

Canonical current files:

- `CURRENT_WORK.md`
- `docs/LONG_TRAINING_PLAN.md`
- `docs/LT2_CHECKPOINT_CROSSPLAY_REVIEW_20260917.md`
- `docs/LT2_POLICY_DRIFT_REVIEW_20260917.md`
- `docs/LT2_STAGE_B_LEARNING_REVIEW_20260917.md`
- `docs/LT2_STAGE_B_RESOURCE_REVIEW_20260917.md`

## Preserved LT2 milestones

Stage A:

- iteration 3000 / 1.8M roots;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B:

- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`;
- all four 2M memories in replacement regime;
- resource gate healthy with zero swap.

## Evidence after Stage B

### Weak fixed opponents

No statistically distinguishable Stage-B-minus-Stage-A improvement or regression across nine ALL/3H/HU comparisons against uniform-legal, passive-caller and jammer families.

### Decision-level policy drift

Material movement is present: overall mean TV 0.041395 and argmax disagreement 12.10%, with much larger postflop/HU movement. Therefore weak-baseline flatness is not equivalent to policy stagnation.

### Contemporary cross-play — first pass

3000 fresh scenarios, seed 20260918.

Primary Stage-B-minus-Stage-A hero delta against the identical deterministic 50/50 A/B opponent mixture:

- ALL: -1.8129 chips/hand, 95% CI [-3.9564,+0.3306];
- 3H: -1.9760, CI [-4.3207,+0.3687];
- HU: -1.6162, CI [-5.4070,+2.1747].

All primary point estimates favor Stage A, but all intervals still include zero. ALL and 3H are borderline.

Additional diagnostics are inconsistent with a simple regression conclusion:

- HU direct B-vs-A: -0.6165, CI [-10.7519,+9.5188];
- 3H invasion difference: +2.8894, CI [-1.2160,+6.9949].

Therefore the correct classification is **borderline/mixed**, not Stage-B improvement and not proven Stage-B regression.

## Immediate decision gate

Do **not** resume training beyond iteration 7500.

Run the exact contemporary cross-play at higher power on an independent seed:

```bash
SPINCORE_CROSSPLAY_SCENARIOS=9000 SPINCORE_CROSSPLAY_SEED=20260919 bash tools/run_lt2_checkpoint_crossplay.sh
```

Interpret the new run independently rather than informally pooling the two reports.

Decision logic:

- repeated coherent Stage-B disadvantage with ALL/3H confidence intervals excluding zero -> investigate training dynamics / AveragePolicy approximation before further roots;
- no disadvantage or reversal -> relative checkpoint evidence remains unresolved, so move emphasis to faithful DeepCrusher / richer external evaluation rather than architecture changes;
- only HU remains unresolved -> run targeted HU evaluation rather than more training.

## Product strength path

Faithful DeepCrusher R8 v22 remains required for canonical strength claims, with HU/3H and later blind/stack/position breakdowns.

## Immediate action

```bash
SPINCORE_CROSSPLAY_SCENARIOS=9000 SPINCORE_CROSSPLAY_SEED=20260919 bash tools/run_lt2_checkpoint_crossplay.sh
```

Wait for `LT2_CHECKPOINT_CROSSPLAY_PASS`, send the resulting cross-play JSON, and do not start additional training first.
