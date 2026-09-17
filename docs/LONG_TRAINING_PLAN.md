# SpinCore — Long-Training Plan

Status: **LT2 STAGE B PASS — SAME-REGIME EXTENSION PAUSED — INITIAL CONTEMPORARY CROSS-PLAY BORDERLINE — 9K FRESH-SEED CONFIRMATION NEXT**
Date: 2026-09-17

## Current state

The continuous learning line has reached:

- LT0: 120k roots — calibration;
- LT1: 1.2M roots — production-shaped milestone;
- LT2 Stage A: 1.8M roots — first policy-reservoir saturation gate;
- LT2 Stage B: 4.5M roots / iteration 7500 — all four 2M memories saturated/replacement;
- paired weak-baseline checkpoint delta — flat/inconclusive;
- policy-drift gate — material movement confirmed;
- first contemporary Stage-A/Stage-B cross-play — borderline/mixed, no demonstrated Stage-B advantage;
- next gate — 9000-scenario fresh-seed contemporary cross-play confirmation.

Read `LT2_CHECKPOINT_CROSSPLAY_REVIEW_20260917.md`, `LT2_POLICY_DRIFT_REVIEW_20260917.md`, `LT2_STAGE_B_LEARNING_REVIEW_20260917.md` and `LT2_STAGE_B_RESOURCE_REVIEW_20260917.md`.

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

## Weak-baseline learning review

Stage A and Stage B were compared on the same 1000 scenarios/deals against uniform-legal, passive-caller and jammer families. All nine paired checkpoint-delta 95% CIs crossed zero. The extra 2.7M roots therefore produced no statistically distinguishable gain or regression under those weak fixed opponents.

## Policy drift

The policies differ materially at the decision-distribution level. Overall mean TV is 0.041395 with 12.10% argmax disagreement across 11,040 identical probe states. Movement is particularly large postflop in HU.

Therefore weak-baseline flatness cannot be treated as policy stagnation.

## Contemporary checkpoint cross-play — initial pass

Method:

- 3000 fresh empirical scenarios;
- seed `20260918`;
- 31 workers;
- same scenario, deal, hero seat and per-seat RNG streams;
- primary opponent environment fixed to the exact same deterministic 50/50 Stage-A/Stage-B mixture;
- additional HU direct and 3H invasion diagnostics.

Primary Stage-B-minus-Stage-A delta:

- ALL: -1.8129 chips/hand, CI [-3.9564,+0.3306];
- 3H: -1.9760, CI [-4.3207,+0.3687];
- HU: -1.6162, CI [-5.4070,+2.1747].

All three point estimates favor Stage A, but all intervals still include zero. ALL and 3H are close to resolving negative.

Additional diagnostics are mixed:

- HU direct B-vs-A: -0.6165, CI [-10.7519,+9.5188];
- 3H invasion B-vs-AA minus A-vs-BB: +2.8894, CI [-1.2160,+6.9949].

This does not support a Stage-B strength claim. It also does not yet prove regression because the primary CIs still cross zero and the invasion diagnostic points in the opposite direction.

## Training decision

Pause same-regime extension beyond iteration 7500.

Do not spend another multi-million-root block while Stage B has failed to demonstrate a relative advantage over Stage A in either weak-baseline or contemporary cross-play evidence.

Also do not change architecture based on the initial 3000-scenario cross-play alone, because it is borderline/mixed.

## Immediate higher-power read-only gate

Repeat the exact contemporary cross-play with triple sample size and an independent seed:

```bash
SPINCORE_CROSSPLAY_SCENARIOS=9000 SPINCORE_CROSSPLAY_SEED=20260919 bash tools/run_lt2_checkpoint_crossplay.sh
```

This performs no training and does not mutate the preserved checkpoints.

Interpretation:

- if Stage B again shows a coherent disadvantage and ALL/3H intervals exclude zero, investigate training dynamics / AveragePolicy approximation before more roots;
- if the disadvantage disappears or reverses, relative checkpoint evidence remains unresolved and the next priority becomes faithful DeepCrusher / richer external benchmarking rather than trainer changes;
- if HU alone remains unresolved, target HU evaluation separately.

Do not informally average the 3000- and 9000-scenario reports. Treat the 9000 fresh-seed run as an independent confirmation.

## DeepCrusher strength path

The faithful DeepCrusher R8 v22 oracle remains the primary external product-strength reference. Once ready, run direct paired chip-EV with HU and 3H separated, then blind/stack/position breakdowns and eventually full-tournament performance.

## Reservoir policy

Do not shrink or enlarge the 2M reservoirs on intuition alone.

## Operational files

- `tools/run_long_training_lt2_stage_b.sh` — completed Stage B launcher;
- `tools/audit_completed_lt2_stage_b.sh` — Stage B postvalidation PASS;
- `tools/run_lt2_stage_b_learning_review.sh` — completed paired weak-baseline review;
- `tools/run_lt2_policy_drift_review.sh` — completed policy-drift gate;
- `tools/evaluate_lt2_checkpoint_crossplay.py` — contemporary policy cross-play evaluator;
- `tools/run_lt2_checkpoint_crossplay.sh` — cross-play launcher, supports scenario/seed overrides;
- `tools/run_lean_functional_training.py` — authoritative trainer.

## Immediate direction

1. Preserve Stage A and Stage B checkpoints.
2. Do not continue training beyond iteration 7500.
3. Pull current `main`.
4. Run `SPINCORE_CROSSPLAY_SCENARIOS=9000 SPINCORE_CROSSPLAY_SEED=20260919 bash tools/run_lt2_checkpoint_crossplay.sh`.
5. Review the fresh-seed 9000-scenario result independently.
6. Then choose between training-dynamics investigation and stronger external benchmarking.
7. Continue the faithful DeepCrusher oracle in parallel.
