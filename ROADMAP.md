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
- Contemporary checkpoint cross-play run 1 (3000 scenarios) — **COMPLETE: BORDERLINE, MILD STAGE-A DIRECTION**.
- Independent cross-play confirmation run 2 (9000 scenarios) — **COMPLETE: SIGN REVERSAL, MILD STAGE-B DIRECTION, STILL INCONCLUSIVE**.
- Same-regime SpinCore extension — **PAUSED AT 4.5M ROOTS**.
- Faithful DeepCrusher R8 v22 external benchmark — **NEXT PRODUCT-STRENGTH GATE**.

Canonical current files:

- `CURRENT_WORK.md`
- `docs/LONG_TRAINING_PLAN.md`
- `docs/LT2_CHECKPOINT_CROSSPLAY_REVIEW_20260917.md`
- `docs/DEEPCRUSHER_BENCHMARK_CONTRACT_20260917.md`
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

Neither checkpoint is discarded. Stage B is not automatically promoted over Stage A until external strength evidence resolves the comparison.

## Evidence after Stage B

### Weak fixed opponents

No statistically distinguishable Stage-B-minus-Stage-A improvement or regression across nine ALL/3H/HU comparisons against uniform-legal, passive-caller and jammer families.

### Decision-level policy drift

Material movement is present: overall mean TV 0.041395 and argmax disagreement 12.10%, with much larger postflop/HU movement. Therefore weak-baseline flatness is not policy stagnation.

### Contemporary cross-play — run 1

3000 fresh scenarios, seed `20260918`.

Primary B-minus-A:

- ALL `-1.8129`, CI `[-3.9564,+0.3306]`;
- 3H `-1.9760`, CI `[-4.3207,+0.3687]`;
- HU `-1.6162`, CI `[-5.4070,+2.1747]`.

All three point estimates favored Stage A; all intervals included zero.

### Contemporary cross-play — independent confirmation

9000 fresh scenarios, seed `20260919`.

Primary B-minus-A:

- ALL `+0.8625`, CI `[-0.4085,+2.1335]`;
- 3H `+0.9152`, CI `[-0.5940,+2.4244]`;
- HU `+0.7990`, CI `[-1.3337,+2.9317]`.

All three primary signs reversed and now mildly favor Stage B, but all intervals again include zero.

Additional confirmation diagnostics:

- HU direct B-vs-A `+1.5503`, CI `[-4.2649,+7.3656]`;
- 3H invasion difference `-2.2202`, CI `[-4.4860,+0.0456]`.

The 3H invasion diagnostic also reversed relative to run 1. The two independent runs therefore do not establish a reproducible checkpoint strength ordering.

Row-level evidence from the 9000-scenario run shows only ~4.19% of primary seat-runs had non-zero paired terminal delta, explaining why the remaining strength signal is sparse and seed-sensitive despite material policy-distribution drift.

## Decision

Do **not** resume same-regime SpinCore training merely to increase root count, and do not continue repeating the same A-vs-B stochastic cross-play.

Current classification:

- training is not stagnant at the policy-distribution level;
- Stage B has not demonstrated reproducible relative strength improvement;
- Stage B has also not demonstrated reproducible regression;
- internal relative evidence is exhausted enough that the next gate should be external.

## Next gate — faithful DeepCrusher R8 v22

Finish and admit the literal DeepCrusher C++ oracle first.

Admission requires faithful OpenPPL rule/order/action semantics and faithful implementation of all relevant library functions rather than approximations. Representative parity probes must pass before the oracle is used for strength claims.

Then benchmark **both Stage A and Stage B** against the exact same DeepCrusher strategy using paired empirical scenarios/deals/seats, with 3H/HU separated wherever faithfully supported.

Decision logic:

- Stage B clearly stronger than Stage A vs DeepCrusher -> current training line remains plausibly productive; consider another bounded continuation from Stage B.
- Stage A clearly stronger than Stage B vs DeepCrusher -> investigate AveragePolicy/training dynamics before more roots.
- Stage A and Stage B externally indistinguishable -> do not spend another long block solely on root count; investigate sensitivity/capacity/optimization through bounded experiments.

See `docs/DEEPCRUSHER_BENCHMARK_CONTRACT_20260917.md`.

## Immediate action

No additional SpinCore compute. Preserve both checkpoints and continue the faithful DeepCrusher R8 v22 C++/OpenPPL-library parity work. Return to SpinCore training only after the external benchmark gate is available and reviewed.
