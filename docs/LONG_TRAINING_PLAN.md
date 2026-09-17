# SpinCore — Long-Training Plan

Status: **LT2 STAGE B PASS — TRAINING PAUSED AT 4.5M ROOTS — HU-JAMMER FAILURE CONFIRMED — FIT AUDIT ACTIVE**
Date: 2026-09-17

## Current state

The continuous learning line has reached:

- LT0: 120k roots — calibration;
- LT1: 1.2M roots — production-shaped milestone;
- LT2 Stage A: 1.8M roots;
- LT2 Stage B: 4.5M roots / iteration 7500;
- all four 2M reservoirs in replacement regime;
- Stage A -> Stage B policy movement is material;
- 30k multi-seed weak-baseline precision gate is complete;
- Stage B is statistically negative versus HU Jammer;
- another blind long-training block is not admitted until training-dynamics diagnostics are reviewed.

Read `LT2_WEAK_BASELINE_VARIANCE_RESULT_20260917.md` and `LT2_TRAINING_DYNAMICS_FIT_AUDIT_20260917.md` first.

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
- 100 Advantage optimizer steps per domain per iteration;
- batch size 1024;
- 4000 AveragePolicy optimizer steps per domain at milestone finalization;
- 31 root workers, one worker numerical thread, 8 parent Torch threads, vectorized batching, production concurrent-fit mode.

## Preserved milestones

LT1: 1.2M roots, SHA256 `beef9bee9439de8d9153450d190e62b0388a678b0778c4185108361f626b4337`.

LT2 Stage A: 1.8M roots / iteration 3000, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

LT2 Stage B: 4.5M roots / iteration 7500, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Stage B final sample state:

- 3H roots 2,452,500;
- HU roots 2,047,500;
- 3H Advantage seen 76,144,669;
- HU Advantage seen 62,622,782;
- 3H AveragePolicy seen 5,549,800;
- HU AveragePolicy seen 2,072,704.

Resource gate passed with zero swap and minimum observed WSL MemAvailable 7.473 GiB.

## What the 30k gate established

The earlier 1000-scenario weak-baseline run was only a pilot. The powered review used six independent 5000-scenario seed blocks per checkpoint, common-random Stage A/B pairing, and Bonferroni simultaneous family-wise 95% intervals across the six primary Stage-B domain/opponent claims.

The predeclared precision target was achieved: maximum primary simultaneous half-width `4.963` chips/hand.

Stage B:

- uniform legal 3H: `+20.102`, simultaneous CI `[+16.372,+23.831]`;
- uniform legal HU: `+21.484`, simultaneous CI `[+16.521,+26.447]`;
- passive caller 3H: `+4.030`, simultaneous CI `[+0.977,+7.082]`;
- passive caller HU: `-0.996`, simultaneous CI `[-4.435,+2.442]`;
- jammer 3H: `+0.859`, simultaneous CI `[-2.628,+4.347]`;
- jammer HU: `-5.141`, simultaneous CI `[-9.078,-1.204]`.

Thus HU Jammer is a confirmed current failure mode. Passive HU and Jammer 3H remain near zero/unresolved at the achieved precision.

The paired Stage-B-minus-Stage-A HU-Jammer delta is `-1.682` chips/hand; after applying the same six-claim simultaneous correction, the interval remains negative at approximately `[-3.143,-0.222]`. The Stage-A -> Stage-B continuation therefore measurably degraded this cell.

This does not mean Stage B is globally worse. Policy drift is material and several cells are strongly positive. It means more roots alone have not produced monotonic useful learning across the weak curriculum.

## Why long training is frozen

A new root block would mix together at least three possible mechanisms:

1. target generation may be noisy or biased in the relevant HU states;
2. the Advantage network may not fit the stored targets well enough after reset + 100 optimizer steps;
3. the AveragePolicy network may not fit its 2M target reservoir adequately after 4000 finalization steps.

Those mechanisms can be separated with read-only or bounded in-memory experiments. They should be separated before paying for another long run.

## Immediate training-dynamics audit

Launcher: `tools/run_lt2_training_dynamics_fit_audit.sh`.

The audit is read only. It deterministically samples Stage A and Stage B Advantage/AveragePolicy reservoirs, separately for 3H/HU, and measures fit of the stored networks to the stored targets.

Advantage metrics:

- legal-action MSE;
- zero-predictor MSE;
- fraction of zero-baseline error removed;
- regret-matching policy TV and argmax agreement;
- target/predicted all-nonpositive fractions;
- sample iteration-age distribution.

AveragePolicy metrics:

- target entropy;
- cross-entropy;
- excess KL;
- TV;
- argmax agreement;
- uniform-legal cross-entropy baseline and fraction of available gap closed;
- sample iteration-age distribution.

There is no arbitrary fit PASS threshold. Stage A/B and 3H/HU comparisons determine the next experiment.

## Bounded branches after the fit audit

If Advantage underfit is implicated, sweep optimizer budget on a cloned in-memory Stage-B model/reservoir with a fixed held-out set. Candidate budgets should include the canonical 100 and larger values selected geometrically; compare held-out MSE/policy-TV rather than training loss alone. Do not mutate the preserved checkpoint.

If AveragePolicy underfit is implicated, sweep policy-fit budget from cloned Stage-B policy state/reservoir and measure held-out KL/TV/argmax agreement. No new roots are needed for that experiment.

If both fits are already strong, move deeper into learning semantics: inspect reservoir age/iteration weighting, HU state/action concentration, target variance, all-nonpositive regret frequency, and whether sampled AveragePolicy trajectories represent the relevant HU responses adequately.

Only after a causal mechanism is identified should a small bounded training experiment be admitted. It must have a predeclared comparison against the preserved Stage B and must not overwrite Stage A or Stage B.

## DeepCrusher placement

DeepCrusher is deferred. It is a sophisticated advanced rules strategy and should be used later, after SpinCore is strong and stable against transparent weak curriculum opponents.

A literal C++ translation may be useful later for speed or auditability, but it is not a prerequisite for eventual benchmarking.

## Immediate direction

1. Preserve Stage A and Stage B.
2. Keep long training stopped at iteration 7500.
3. Run `bash tools/run_lt2_training_dynamics_fit_audit.sh`.
4. Review `SpinCore_LT2_training_fit_audit.json`.
5. Choose one bounded causal experiment from the observed fit evidence.
6. Do not return to long-root scaling or DeepCrusher yet.
