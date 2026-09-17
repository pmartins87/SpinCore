# SpinCore Current Work

Date: 2026-09-17
Status: **LT2 STAGE B PASS — 4.5M ROOTS — 30K WEAK-BASELINE GATE RESOLVED HU-JAMMER NEGATIVE — TRAINING-DYNAMICS FIT AUDIT NEXT**

## Active source of truth

Read before new compute:

- `docs/LT2_WEAK_BASELINE_VARIANCE_RESULT_20260917.md`
- `docs/LT2_TRAINING_DYNAMICS_FIT_AUDIT_20260917.md`
- `docs/LT2_VARIANCE_AND_WEAK_BASELINE_GATE_20260917.md`
- `docs/LT2_CHECKPOINT_CROSSPLAY_REVIEW_20260917.md`
- `docs/LT2_POLICY_DRIFT_REVIEW_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`

Preserve Stage A and Stage B. Do not continue long training beyond iteration 7500 until the training-dynamics fit audit is reviewed.

## Preserved checkpoints

Stage A: iteration 3000 / 1.8M roots, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Stage B has all four 2M memories in replacement regime and healthy resource status.

## 30k weak-baseline result

Six independent 5000-scenario seeds produced 30,000 scenarios per checkpoint. The predeclared simultaneous precision target was met: maximum primary family-wise 95% half-width `4.963` chips/hand.

Stage B primary simultaneous family-wise 95% results:

- `UNIFORM_LEGAL` 3H: `+20.102`, CI `[+16.372,+23.831]` — **POSITIVE**;
- `UNIFORM_LEGAL` HU: `+21.484`, CI `[+16.521,+26.447]` — **POSITIVE**;
- `PASSIVE_CALLER` 3H: `+4.030`, CI `[+0.977,+7.082]` — **POSITIVE**;
- `PASSIVE_CALLER` HU: `-0.996`, CI `[-4.435,+2.442]` — **UNRESOLVED**;
- `JAMMER` 3H: `+0.859`, CI `[-2.628,+4.347]` — **UNRESOLVED**;
- `JAMMER` HU: `-5.141`, CI `[-9.078,-1.204]` — **NEGATIVE**.

Therefore the earlier apparent HU-Jammer weakness was not merely small-sample variance. Stage B has a statistically resolved negative chip EV against this weak baseline family.

## What Stage A -> Stage B changed

Common-random paired A/B evidence is more sensitive than independent raw EV.

Ordinary paired 95% CIs are negative for Stage B relative to Stage A in Jammer HU, Jammer 3H, and Passive HU. After applying the same six-claim Bonferroni simultaneous correction to the paired comparisons, only Jammer HU remains resolved:

- Jammer HU Stage B minus Stage A: `-1.682` chips/hand;
- simultaneous family-wise 95% CI approximately `[-3.143,-0.222]`.

Thus the extra Stage-A -> Stage-B training **worsened HU-vs-Jammer performance in a statistically robust way**. Other cells should remain classified unresolved rather than overinterpreted.

This result coexists with the previously measured material policy drift. Training is changing the policy, but the change is not uniformly beneficial.

## Current diagnostic question

Do not add roots yet. Determine whether the training approximation is underfitting its own memories before changing architecture or sampling.

Known current mechanics:

- each domain's Advantage network is reset every iteration;
- it is then fitted for 100 optimizer steps from the accumulated Advantage reservoir;
- AveragePolicy is fitted at milestone finalization for 4000 optimizer steps;
- Stage B reservoirs are at 2M capacity.

These are hypotheses, not declared bugs.

## Immediate read-only fit audit

Run:

```bash
bash tools/run_lt2_training_dynamics_fit_audit.sh
```

The audit uses deterministic samples from Stage A and Stage B Advantage/AveragePolicy reservoirs and measures, separately for 3H and HU:

- memory capacity/seen counts and iteration-age sample distribution;
- Advantage MSE versus a zero predictor;
- regret-matching policy TV and argmax agreement induced by Advantage predictions;
- AveragePolicy target entropy, model cross-entropy, excess KL, TV and argmax agreement;
- optimizer/reset counters.

There is no arbitrary pass threshold. The observed fit quality determines the next bounded experiment.

## Decision after fit audit

- poor Advantage fit, especially Stage B HU -> run a controlled in-memory optimizer-budget sweep before changing anything else;
- poor AveragePolicy fit -> run a controlled policy-fit budget sweep from the preserved checkpoint, without new roots;
- both fits already good -> investigate target-generation / learning semantics and HU-specific state/action concentration rather than merely increasing optimizer steps;
- do not resume a long root block until one of these mechanisms is resolved.

DeepCrusher remains a later advanced benchmark, not a current dependency.

## Immediate user action

Pull current `main` and run `bash tools/run_lt2_training_dynamics_fit_audit.sh`. Send `SpinCore_LT2_training_fit_audit.json`. Do not resume long training first.
