# SpinCore Roadmap — active state 2026-09-17

## Active status

- LT0 calibration — **DONE**: 120k roots.
- LT1 production-shaped milestone — **DONE**: 1.2M roots.
- LT1 physical fit optimization — **PASS**: 31 root workers, 8 parent Torch threads, vectorized batching.
- LT2 Stage A — **PASS**: 1.8M roots.
- Concurrent-fit production parity — **PASS**.
- LT2 Stage B — **PASS**: 4.5M roots / iteration 7500.
- Stage B resource gate — **PASS**: all four 2M memories in replacement regime, zero swap.
- Stage A -> Stage B policy drift — **MATERIAL MOVEMENT CONFIRMED**.
- Checkpoint cross-play 3000 + independent 9000 scenarios — **SIGN-UNSTABLE / NO REPRODUCIBLE ORDERING**.
- 30k multi-seed weak-baseline variance gate — **COMPLETE; PRECISION TARGET MET; HU JAMMER CONFIRMED NEGATIVE**.
- Training-dynamics fit audit — **NEXT**.
- DeepCrusher — **DEFERRED TO LATER ADVANCED BENCHMARK; NOT CURRENT GATE**.

Canonical current files:

- `CURRENT_WORK.md`
- `docs/LT2_WEAK_BASELINE_VARIANCE_RESULT_20260917.md`
- `docs/LT2_TRAINING_DYNAMICS_FIT_AUDIT_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`
- `docs/LT2_CHECKPOINT_CROSSPLAY_REVIEW_20260917.md`
- `docs/LT2_POLICY_DRIFT_REVIEW_20260917.md`

## Preserved checkpoints

Stage A: iteration 3000 / 1.8M roots, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Keep both unchanged.

## Weak-baseline gate result

The 30k evaluation used six independent 5000-scenario seed blocks per checkpoint and met the predeclared simultaneous precision target: maximum primary family-wise 95% half-width `4.963` chips/hand.

Stage B primary simultaneous family-wise 95% results:

- uniform legal 3H: `+20.102`, CI `[+16.372,+23.831]` — positive;
- uniform legal HU: `+21.484`, CI `[+16.521,+26.447]` — positive;
- passive caller 3H: `+4.030`, CI `[+0.977,+7.082]` — positive;
- passive caller HU: `-0.996`, CI `[-4.435,+2.442]` — unresolved;
- jammer 3H: `+0.859`, CI `[-2.628,+4.347]` — unresolved;
- jammer HU: `-5.141`, CI `[-9.078,-1.204]` — **negative**.

The small-pilot HU-Jammer weakness was therefore not merely variance.

The paired Stage-B-minus-Stage-A HU-Jammer delta is `-1.682` chips/hand. Using the same six-claim Bonferroni correction, its simultaneous interval remains negative at approximately `[-3.143,-0.222]`. Thus the extra Stage-A -> Stage-B training measurably worsened this particular weak-opponent cell, even though other parts of the policy changed without a resolved strength ordering.

## Why more roots are paused

The policy is moving materially, so the problem is not simple stagnation. But more samples did not uniformly improve weak-opponent strength, and one HU failure mode became statistically worse. Another long root block would therefore confound sample quantity with approximation quality.

Current mechanics worth testing rather than assuming faulty:

- Advantage network reset every iteration;
- 100 Advantage optimizer steps per domain per iteration;
- 4000 AveragePolicy optimizer steps only at milestone finalization;
- 2M-capacity reservoirs with iteration-weighted losses.

## Immediate gate — read-only fit audit

Run `tools/run_lt2_training_dynamics_fit_audit.sh`.

It samples stored Stage A/B Advantage and AveragePolicy memories and measures model fit to stored targets, separately for 3H/HU. It performs no training and does not mutate checkpoints.

Important metrics include:

- Advantage MSE versus a zero predictor;
- Advantage regret-matching policy TV and argmax agreement;
- AveragePolicy target entropy, cross-entropy, excess KL, TV and argmax agreement;
- reservoir seen counts and sampled iteration-age distribution;
- optimizer/reset counters.

There is no arbitrary PASS threshold. The observed mechanism determines the next bounded experiment.

## Branch after fit audit

If Stage B HU Advantage fit is clearly poor, run a controlled optimizer-budget sweep on the preserved Stage-B memory before collecting new roots.

If AveragePolicy fit is poor, run a controlled policy-fit budget sweep from the preserved checkpoint, again without new roots.

If both fits are already strong, investigate target generation, reservoir weighting/age, state/action concentration and HU-specific learning semantics rather than increasing fit budgets blindly.

Only after the current weak-opponent failure is understood and corrected should a new bounded root block be admitted.

## DeepCrusher placement

DeepCrusher remains a later advanced reference. A literal C++ transcription can be useful for speed/auditability but is not logically required for eventual benchmarking. It is not a dependency for the present diagnosis.

## Immediate action

```bash
bash tools/run_lt2_training_dynamics_fit_audit.sh
```

Do not resume long training beyond iteration 7500 until the fit-audit report is reviewed.
