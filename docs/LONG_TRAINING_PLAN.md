# SpinCore — Long-Training Plan

Status: **LT2 STAGE B PASS — TRAINING FROZEN AT 4.5M ROOTS — MATERIAL POLICY DRIFT BUT NO REPRODUCIBLE CHECKPOINT STRENGTH ORDERING — DEEPCRUSHER EXTERNAL GATE NEXT**
Date: 2026-09-17

## Current state

The continuous learning line has reached:

- LT0: 120k roots — calibration;
- LT1: 1.2M roots — production-shaped milestone;
- LT2 Stage A: 1.8M roots — first policy-reservoir saturation gate;
- LT2 Stage B: 4.5M roots / iteration 7500 — all four 2M memories saturated/replacement;
- paired weak-baseline checkpoint delta — flat/inconclusive;
- policy-drift gate — material movement confirmed;
- contemporary Stage-A/Stage-B cross-play run 1 — mild Stage-A direction, inconclusive;
- independent 9000-scenario cross-play confirmation — sign reversal to mild Stage-B direction, still inconclusive;
- next gate — faithful DeepCrusher R8 v22 external benchmark of both preserved checkpoints.

Read `LT2_CHECKPOINT_CROSSPLAY_REVIEW_20260917.md`, `DEEPCRUSHER_BENCHMARK_CONTRACT_20260917.md`, `LT2_POLICY_DRIFT_REVIEW_20260917.md`, `LT2_STAGE_B_LEARNING_REVIEW_20260917.md` and `LT2_STAGE_B_RESOURCE_REVIEW_20260917.md`.

## Core training contract

Current functional line:

- empirical SpinGo 3H/HU/blind/stack sampling;
- WTA chip-EV utility scaled by 1500;
- SPNNIV1 frozen-control representation;
- mature legacy action vocabulary;
- external-sampling Deep CFR with repaired all-nonpositive regret fallback;
- separate 3H and HU brains;
- sampled AveragePolicy trajectories;
- 2,000,000-sample reservoir capacity per memory per domain;
- 600 roots per iteration;
- 100 Advantage optimizer steps per domain per iteration;
- batch size 1024;
- 4000 AveragePolicy optimizer steps per domain at milestone finalization.

Admitted Ryzen execution profile:

- 31 root workers;
- one numerical-library thread per root worker;
- 8 parent Torch threads;
- vectorized batch construction;
- production `concurrent_fit` iteration mode.

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

All four 2M memories are in replacement regime. Resource gate passed with zero swap and minimum observed WSL MemAvailable 7.473 GiB.

Preserve Stage A and Stage B. Do not assume Stage B is stronger simply because it is later.

## Weak-baseline learning review

Stage A and Stage B were compared on the same 1000 scenarios/deals against uniform-legal, passive-caller and jammer families. All nine paired checkpoint-delta 95% CIs crossed zero. The extra 2.7M roots therefore produced no statistically distinguishable gain or regression under those weak fixed opponents.

## Policy drift

The policies differ materially at the decision-distribution level. Overall mean TV is 0.041395 with 12.10% argmax disagreement across 11,040 identical probe states. Movement is particularly large postflop in HU.

Therefore weak-baseline flatness cannot be treated as policy stagnation.

## Contemporary checkpoint cross-play — run 1

3000 scenarios, seed `20260918`.

Primary Stage-B-minus-Stage-A delta against the identical deterministic 50/50 A/B opponent mixture:

- ALL `-1.8129`, CI `[-3.9564,+0.3306]`;
- 3H `-1.9760`, CI `[-4.3207,+0.3687]`;
- HU `-1.6162`, CI `[-5.4070,+2.1747]`.

Additional diagnostics:

- HU direct `-0.6165`, CI `[-10.7519,+9.5188]`;
- 3H invasion difference `+2.8894`, CI `[-1.2160,+6.9949]`.

The primary point estimates favored Stage A, but all intervals included zero and the invasion diagnostic pointed the opposite way.

## Contemporary checkpoint cross-play — independent confirmation

9000 scenarios, seed `20260919`; 4918 3H / 4082 HU.

Primary Stage-B-minus-Stage-A delta:

- ALL `+0.8625`, CI `[-0.4085,+2.1335]`;
- 3H `+0.9152`, CI `[-0.5940,+2.4244]`;
- HU `+0.7990`, CI `[-1.3337,+2.9317]`.

Additional diagnostics:

- HU direct `+1.5503`, CI `[-4.2649,+7.3656]`;
- 3H invasion difference `-2.2202`, CI `[-4.4860,+0.0456]`.

The higher-power independent run reversed all three primary signs relative to run 1 and still did not exclude zero. The invasion diagnostic also reversed direction.

Row-level evidence shows only about 4.19% of 22,918 primary seat-runs produced a non-zero paired terminal chip delta. The paired common-random-number design cancels most trajectories exactly, but the remaining rare divergent trajectories carry large positive/negative outcomes. That makes further repetitions of the same A-vs-B stochastic cross-play a low-value use of evaluation compute.

## Training decision

Freeze same-regime extension beyond iteration 7500 / 4.5M roots.

Evidence now supports all of the following simultaneously:

- training continues to move the AveragePolicy materially;
- weak fixed opponents do not resolve whether that movement helps;
- contemporary checkpoint cross-play does not provide a reproducible A/B strength ordering across independent seeds;
- there is no defensible basis yet for either more blind root count or an architecture change.

Do not spend another multi-million-root block merely because resources are healthy. Do not repeatedly rerun the same cross-play seeking significance.

## Next product-strength gate — faithful DeepCrusher R8 v22

The DeepCrusher oracle must pass source-faithfulness admission before use:

- structural OpenPPL rule ordering/priority preserved;
- all relevant library symbols implemented from their real definitions, including functions such as `AmountToCall`;
- action sizing/all-in conversion semantics preserved;
- representative parity probes against the frozen source pass;
- intentional divergences documented.

After admission, benchmark **both Stage A and Stage B** against the exact same DeepCrusher policy with paired empirical scenarios, deal seeds, hero-seat rotation and row-level evidence.

Required primary outputs:

- Stage A vs DeepCrusher chip EV;
- Stage B vs DeepCrusher chip EV;
- paired Stage-B-minus-Stage-A external-reference delta;
- 3H and HU separated wherever DeepCrusher faithfully supports them;
- scenario-clustered uncertainty.

Then add blind/effective-stack/position breakdowns only after the primary result is stable.

Decision logic:

- Stage B clearly stronger than Stage A vs DeepCrusher -> consider another bounded continuation from Stage B and re-evaluate afterward;
- Stage A clearly stronger -> investigate AveragePolicy/training dynamics before more roots;
- externally indistinguishable -> investigate representation/capacity/optimizer/reservoir dynamics only through bounded controlled experiments, not another long blind run.

See `docs/DEEPCRUSHER_BENCHMARK_CONTRACT_20260917.md`.

## Reservoir policy

Do not shrink or enlarge the 2M reservoirs on intuition alone. Stage B established a healthy saturated operating point with zero swap.

## Operational files

- `tools/run_long_training_lt2_stage_b.sh` — completed Stage B launcher;
- `tools/audit_completed_lt2_stage_b.sh` — Stage B postvalidation PASS;
- `tools/run_lt2_stage_b_learning_review.sh` — completed paired weak-baseline review;
- `tools/run_lt2_policy_drift_review.sh` — completed policy-drift gate;
- `tools/evaluate_lt2_checkpoint_crossplay.py` — contemporary policy cross-play evaluator;
- `tools/run_lt2_checkpoint_crossplay.sh` — completed cross-play launcher with scenario/seed overrides;
- `tools/run_lean_functional_training.py` — authoritative trainer.

## Immediate direction

1. Preserve Stage A and Stage B checkpoints.
2. Keep SpinCore training stopped at iteration 7500 / 4.5M roots.
3. Do not spend more evaluation compute on repeated A-vs-B stochastic cross-play by default.
4. Finish the faithful DeepCrusher R8 v22 C++/OpenPPL-library parity work.
5. Admit DeepCrusher only after representative source-parity checks pass.
6. Benchmark both Stage A and Stage B against the same external oracle.
7. Use that result to decide whether to continue Stage B, investigate training dynamics, or run bounded architecture/optimization experiments.
