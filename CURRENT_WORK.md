# SpinCore Current Work

Date: 2026-09-17
Status: **LT2 STAGE B PASS — 4.5M ROOTS — MATERIAL POLICY DRIFT — INITIAL CROSS-PLAY BORDERLINE / MIXED — 9K FRESH-SEED CONFIRMATION NEXT**

## Active source of truth

Read before new compute:

- `docs/LT2_CHECKPOINT_CROSSPLAY_REVIEW_20260917.md`
- `docs/LT2_POLICY_DRIFT_REVIEW_20260917.md`
- `docs/LT2_STAGE_B_LEARNING_REVIEW_20260917.md`
- `docs/LT2_STAGE_B_RESOURCE_REVIEW_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`

Preserve all completed checkpoints. Do not continue training beyond iteration 7500 until the higher-power fresh-seed cross-play confirmation is reviewed.

## Preserved checkpoints

LT2 Stage A:

- iteration 3000 / 1.8M roots;
- `/home/rz9/spincore_lean_functional/runs/long_training_lt2/20260916_015559/checkpoint.pt`;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

LT2 Stage B:

- iteration 7500 / 4.5M roots;
- `/home/rz9/spincore_lean_functional/runs/long_training_lt2_stage_b/20260917_004911/checkpoint.pt`;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Stage B state

- 4,500,000 total roots;
- 3H roots 2,452,500;
- HU roots 2,047,500;
- 3H AveragePolicy seen 5,549,800;
- HU AveragePolicy seen 2,072,704;
- all four 2M reservoirs in saturation/replacement regime;
- final checkpoint 2.623474 GiB;
- min observed WSL MemAvailable 7.473 GiB;
- max observed swap 0 GiB.

Resource status is healthy.

## Weak-baseline review

Stage A and Stage B were evaluated on the same 1000 empirical scenarios/deals against uniform-legal, passive-caller and jammer families. All nine Stage-B-minus-Stage-A 95% CIs included zero.

Result: no statistically distinguishable Stage-B improvement or regression under the weak fixed-opponent sentinel.

## Policy drift

The policies were queried on 11,040 identical decision states from 3000 checkpoint-independent probe scenarios.

Overall:

- mean TV 0.041395;
- median TV 0.032119;
- p95 TV 0.108574;
- 26.77% of decisions TV >= 0.05;
- argmax disagreement 12.10%.

Movement is especially strong postflop in HU. Therefore the flat weak-baseline result is not policy stagnation.

## Initial contemporary checkpoint cross-play — COMPLETE / BORDERLINE

3000 fresh scenarios, seed `20260918`, 1640 3H and 1360 HU.

Primary Stage-B-minus-Stage-A result against the identical deterministic 50/50 Stage-A/Stage-B opponent mixture:

| Domain | Delta chips/hand | 95% CI |
|---|---:|---:|
| ALL | -1.8129 | [-3.9564,+0.3306] |
| 3H | -1.9760 | [-4.3207,+0.3687] |
| HU | -1.6162 | [-5.4070,+2.1747] |

All primary point estimates favor Stage A, but all confidence intervals still include zero. ALL and 3H are close to excluding zero.

Additional diagnostics are mixed:

- HU direct B-vs-A: -0.6165, CI [-10.7519,+9.5188];
- 3H invasion difference B-vs-AA minus A-vs-BB: +2.8894, CI [-1.2160,+6.9949].

Therefore Stage B has **not demonstrated relative strength improvement**, but a regression is not yet established either. The 3H invasion diagnostic points opposite the primary mixture point estimate.

See `docs/LT2_CHECKPOINT_CROSSPLAY_REVIEW_20260917.md`.

## Immediate finite gate — 9000-scenario fresh-seed cross-play confirmation

Do not train more yet.

Run the exact same cross-play at triple sample size on an independent fresh seed:

```bash
SPINCORE_CROSSPLAY_SCENARIOS=9000 SPINCORE_CROSSPLAY_SEED=20260919 bash tools/run_lt2_checkpoint_crossplay.sh
```

This is read-only and performs no training/checkpoint mutation.

Decision logic after the 9000-scenario confirmation:

- coherent Stage-B disadvantage, especially if ALL/3H CIs exclude zero -> pause same-regime training and investigate training dynamics / AveragePolicy approximation before more roots;
- no disadvantage or direction reversal -> checkpoint-vs-checkpoint strength remains unresolved; prioritize faithful DeepCrusher / richer external benchmark instead of changing the trainer from noisy relative evidence;
- HU unresolved while ALL/3H resolve -> target HU evaluation separately rather than spending training compute.

Do not average the 3000- and 9000-scenario reports informally; interpret the 9000 fresh-seed run as an independent confirmation.

## Immediate user action

Pull current `main` and run:

```bash
SPINCORE_CROSSPLAY_SCENARIOS=9000 SPINCORE_CROSSPLAY_SEED=20260919 bash tools/run_lt2_checkpoint_crossplay.sh
```

Wait for `LT2_CHECKPOINT_CROSSPLAY_PASS`, then send the new `SpinCore_LT2A_to_LT2B_crossplay.json`. Do not resume training first.
