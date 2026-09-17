# SpinCore — Long-Training Plan

Status: **LT2 STAGE B PASS — ROOT TRAINING PAUSED AT 4.5M — HU-JAMMER FAILURE CONFIRMED — HELD-OUT FIT-BUDGET SWEEP ACTIVE**
Date: 2026-09-17

## Current state

The continuous learning line has reached:

- LT0: 120k roots;
- LT1: 1.2M roots;
- LT2 Stage A: 1.8M roots;
- LT2 Stage B: 4.5M roots / iteration 7500;
- all four 2M reservoirs in replacement regime;
- Stage A -> Stage B policy movement is material;
- 30k weak-baseline precision gate complete;
- Stage B HU Jammer is statistically negative and worsened relative to Stage A;
- stored-target fit audit complete;
- immediate next experiment is an in-memory held-out optimizer-budget sweep, not more roots.

Read `LT2_TRAINING_DYNAMICS_FIT_RESULT_20260917.md` first.

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
- 31 root workers, one worker numerical thread, 8 parent Torch threads, vectorized batching, production concurrent-fit mode.

## Preserved milestones

LT1: 1.2M roots, SHA256 `beef9bee9439de8d9153450d190e62b0388a678b0778c4185108361f626b4337`.

LT2 Stage A: 1.8M roots / iteration 3000, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

LT2 Stage B: 4.5M roots / iteration 7500, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Stage B sample state:

- 3H roots 2,452,500;
- HU roots 2,047,500;
- 3H Advantage seen 76,144,669;
- HU Advantage seen 62,622,782;
- 3H AveragePolicy seen 5,549,800;
- HU AveragePolicy seen 2,072,704.

## Strength result that freezes root scaling

The statistically powered 30k gate established:

- Stage B HU Jammer `-5.141` chips/hand, simultaneous family-wise 95% CI `[-9.078,-1.204]`;
- Stage B minus Stage A HU Jammer `-1.682`, simultaneous six-claim CI about `[-3.143,-0.222]`.

Therefore the extra 2.7M roots from Stage A to Stage B measurably degraded this specific weak-opponent HU cell. More roots cannot be admitted merely because the policy continues to move.

## Stored-target fit audit

A read-only 25k-per-memory audit measured the stored networks against stored reservoir targets.

Stage B weighted results:

- 3H Advantage: zero-baseline error removed `9.98%`, induced-policy TV `0.5969`, argmax `33.56%`;
- HU Advantage: `14.17%`, TV `0.6058`, argmax `26.40%`;
- 3H AveragePolicy: uniform-to-target gap closed `12.98%`, KL `0.6323`, TV `0.4286`, argmax `51.46%`;
- HU AveragePolicy: `16.27%`, KL `0.6387`, TV `0.4318`, argmax `49.20%`.

Stage A is broadly similar. Stage-B HU AveragePolicy is modestly worse on several metrics but there is no evidence of a unique catastrophic Stage-B approximation collapse.

These absolute residuals cannot by themselves diagnose optimizer underfit because Advantage targets contain external-sampling variance and AveragePolicy contains historical iteration-varying targets. A held-out learning curve is required.

## Scale observation

A freshly reset Advantage model receives `100 x 1024 = 102,400` sample draws from a 2M reservoir. Under a simple independent-draw approximation, expected unique coverage is about 99.8k items, approximately 5.0% of the reservoir, before that iteration's behavior policy is generated. This does not prove 100 is inadequate; it motivates testing the budget directly.

AveragePolicy has not had only 4000 lifetime steps. Counters show 8000 cumulative at Stage A and 12000 at Stage B; 4000 is the increment at each milestone finalization.

## Immediate bounded experiment

Launcher: `tools/run_lt2_stage_b_fit_budget_sweep.sh`.

The experiment uses Stage B only and never writes the source checkpoint. It creates a deterministic 25k holdout for each memory and excludes held-out items from all optimization sampling.

Advantage learning curve:

- one fresh deterministic reset;
- cumulative budgets `0,25,50,100,200,400,800,1600`;
- production reference = 100 steps;
- metrics: held-out MSE, zero-baseline fraction removed, regret-matching policy TV, argmax agreement.

AveragePolicy learning curve:

- start from the stored Stage-B model and optimizer;
- cumulative additional budgets `0,1000,2000,4000,8000`;
- production scale reference = +4000 finalization steps;
- metrics: held-out CE/KL/TV/argmax and uniform-gap fraction.

The budgets are geometric diagnostics rather than pass thresholds. The causal signal is whether held-out error continues to decrease beyond the current production budget and whether the curve approaches a plateau.

## Decision branches

If Advantage continues improving substantially beyond 100, test a larger Advantage budget in a small isolated continuation with no checkpoint overwrite and a predeclared weak-baseline comparison.

If AveragePolicy improves substantially with additional fitting, create a policy-only refit candidate first, without new roots, and benchmark it against the same weak baselines.

If either learning curve plateaus while residual error remains large, optimizer budget is not the main bottleneck. Move to model capacity, SPNNIV1 representation/aliasing, target variance and reservoir weighting diagnostics.

If both plateau early, inspect HU target generation and state/action concentration before changing training volume.

Only after a causal mechanism is identified and an isolated fix shows improvement can another root block be admitted.

## DeepCrusher placement

DeepCrusher remains deferred until the weak-opponent curriculum is strong and stable. A literal C++ translation may help later but is not a current dependency.

## Immediate direction

1. Preserve Stage A and Stage B.
2. Keep root training stopped at iteration 7500.
3. Run `bash tools/run_lt2_stage_b_fit_budget_sweep.sh`.
4. Review `SpinCore_LT2_stage_b_fit_budget_sweep.json`.
5. Choose the next isolated intervention from the held-out curves.
6. Do not scale roots or move to DeepCrusher yet.
