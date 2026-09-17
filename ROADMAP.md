# SpinCore Roadmap — active state 2026-09-17

## Active status

- LT0 calibration — **DONE**: 120k roots.
- LT1 production-shaped milestone — **DONE**: 1.2M roots.
- LT1 physical fit optimization — **PASS**: 31 root workers, 8 parent Torch threads, vectorized batching.
- LT2 Stage A — **PASS**: 1.8M roots.
- Concurrent-fit production parity — **PASS**.
- LT2 Stage B — **PASS**: 4.5M roots / iteration 7500.
- Stage B resource gate — **PASS**: all four 2M memories saturated/replacement, zero swap.
- Stage A -> Stage B weak-baseline 1000-scenario review — **INCONCLUSIVE / LOW PRECISION**.
- Stage A -> Stage B policy drift — **MATERIAL MOVEMENT CONFIRMED**.
- Checkpoint cross-play 3000 + independent 9000 scenarios — **SIGN-UNSTABLE / NO REPRODUCIBLE ORDERING**.
- 30k multi-seed weak-baseline variance gate — **NEXT**.
- DeepCrusher — **DEFERRED TO LATER ADVANCED BENCHMARK; NOT CURRENT GATE**.

Canonical current files:

- `CURRENT_WORK.md`
- `docs/LT2_VARIANCE_AND_WEAK_BASELINE_GATE_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`
- `docs/LT2_CHECKPOINT_CROSSPLAY_REVIEW_20260917.md`
- `docs/LT2_POLICY_DRIFT_REVIEW_20260917.md`

## Preserved checkpoints

Stage A: iteration 3000 / 1.8M roots, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Keep both.

## Why the previous DeepCrusher direction was wrong

DeepCrusher is a very advanced rules strategy. It is useful later as a demanding external reference, but it is not a coherent early pass/fail target while SpinCore has not yet demonstrated statistically stable superiority over transparent weak opponents.

The C++ literal-transcription project is optional for future benchmarking. Any future DeepCrusher benchmark only requires a demonstrably faithful execution path; it does not logically require that translation to be completed first.

## What the data currently supports

The 1000-scenario Stage-B weak-baseline pilot has wide uncertainty. Raw Stage-B 95% half-widths were roughly 9–20 chips/hand depending on domain/opponent. Apparent HU losses versus passive caller and jammer therefore remain compatible with sampling variance.

Policy drift is real: Stage A and Stage B differ materially, especially postflop/HU. Therefore the problem is not simply “training stopped changing the policy.”

Checkpoint cross-play is not a reliable strength gate yet. The 3000-scenario run mildly favored Stage A; the independent 9000-scenario run reversed all three primary signs to mildly favor Stage B. Neither excluded zero. This is exactly the kind of result that must be treated as variance/seed sensitivity rather than a pass/fail event.

## Immediate curriculum gate — weak opponents at adequate precision

Use `tools/run_lt2_weak_baseline_variance_review.sh`.

Design:

- six independent seeds;
- 5000 scenarios each;
- 30,000 scenarios per checkpoint;
- Stage A/B common-random pairing within seed;
- 31 workers;
- no training.

Primary claims are Stage-B raw chip EV versus:

- uniform legal, 3H;
- uniform legal, HU;
- passive caller, 3H;
- passive caller, HU;
- jammer, 3H;
- jammer, HU.

All six use simultaneous family-wise 95% confidence intervals. No arbitrary chip-EV pass number is imposed: positive means the simultaneous lower bound is above zero; negative means the upper bound is below zero; otherwise the result is unresolved.

30k is derived from the pilot variance: it is the bounded sample needed to target roughly <=5 chips/hand worst-case simultaneous half-width, enough to resolve whether the pilot's apparent ~8 to ~12 chip HU losses are real.

## Branch after the 30k gate

If all six are clearly positive, SpinCore has cleared the weak-opponent curriculum gate and another bounded continuation can be considered.

If one or more are negative, or remain essentially near zero once precision is adequate, investigate training dynamics before adding roots. First targets:

- Advantage reset every iteration + only 100 optimizer steps from random initialization;
- held-out Advantage-memory loss versus larger fit budgets;
- AveragePolicy fit quality after 4000 final steps on a 2M reservoir;
- reservoir composition/age and weighting;
- 3H/HU-specific concentration.

These are testable hypotheses, not assumed bugs.

## Later product-strength path

Only after weak-baseline strength is solid should stronger references such as DeepCrusher become useful. At that point choose the most reliable faithful DeepCrusher execution route; a literal C++ transcription may help speed/auditability but is not mandatory by definition.

## Immediate action

```bash
bash tools/run_lt2_weak_baseline_variance_review.sh
```

Do not resume training until the resulting 30k multi-seed report is reviewed.
