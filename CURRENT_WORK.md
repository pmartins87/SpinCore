# SpinCore Current Work

Date: 2026-09-17
Status: **LT2 STAGE B PASS — 4.5M ROOTS — MATERIAL POLICY DRIFT — TWO CROSS-PLAY RUNS INCONCLUSIVE / SIGN-UNSTABLE — TRAINING FROZEN PENDING FAITHFUL DEEPCRUSHER BENCHMARK**

## Active source of truth

Read before new compute:

- `docs/LT2_CHECKPOINT_CROSSPLAY_REVIEW_20260917.md`
- `docs/DEEPCRUSHER_BENCHMARK_CONTRACT_20260917.md`
- `docs/LT2_POLICY_DRIFT_REVIEW_20260917.md`
- `docs/LT2_STAGE_B_LEARNING_REVIEW_20260917.md`
- `docs/LT2_STAGE_B_RESOURCE_REVIEW_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`

Preserve all completed checkpoints. Do not continue SpinCore training beyond iteration 7500 until the external benchmark gate below is available and reviewed.

## Preserved checkpoints

LT2 Stage A:

- iteration 3000 / 1.8M roots;
- `/home/rz9/spincore_lean_functional/runs/long_training_lt2/20260916_015559/checkpoint.pt`;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

LT2 Stage B:

- iteration 7500 / 4.5M roots;
- `/home/rz9/spincore_lean_functional/runs/long_training_lt2_stage_b/20260917_004911/checkpoint.pt`;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Do not overwrite either. Stage B is not automatically promoted over Stage A for strength claims.

## Stage B state

- 4,500,000 total roots;
- 3H roots 2,452,500;
- HU roots 2,047,500;
- 3H AveragePolicy seen 5,549,800;
- HU AveragePolicy seen 2,072,704;
- all four 2M reservoirs in replacement regime;
- final checkpoint 2.623474 GiB;
- minimum observed WSL MemAvailable 7.473 GiB;
- maximum observed swap 0 GiB.

Resource status is healthy.

## Evidence after Stage B

### Weak fixed opponents

Stage A and Stage B were compared on the same 1000 empirical scenarios/deals against uniform-legal, passive-caller and jammer families. All nine paired Stage-B-minus-Stage-A 95% CIs included zero.

Result: no statistically distinguishable improvement or regression under the weak fixed-opponent sentinel.

### Policy drift

Across 11,040 identical probe states from 3000 independent scenarios:

- overall mean TV 0.041395;
- median TV 0.032119;
- p95 TV 0.108574;
- 26.77% of decisions TV >= 0.05;
- argmax disagreement 12.10%.

Movement is especially strong postflop in HU. Therefore weak-baseline flatness is not policy stagnation.

### Contemporary checkpoint cross-play — two independent runs

Run 1: 3000 scenarios, seed `20260918`.

Primary B-minus-A:

- ALL `-1.8129`, CI `[-3.9564,+0.3306]`;
- 3H `-1.9760`, CI `[-4.3207,+0.3687]`;
- HU `-1.6162`, CI `[-5.4070,+2.1747]`.

Run 2: 9000 scenarios, independent seed `20260919`.

Primary B-minus-A:

- ALL `+0.8625`, CI `[-0.4085,+2.1335]`;
- 3H `+0.9152`, CI `[-0.5940,+2.4244]`;
- HU `+0.7990`, CI `[-1.3337,+2.9317]`.

The higher-power fresh-seed run reversed all three primary signs and still did not exclude zero. Additional HU-direct and 3H-invasion diagnostics were also broad and directionally unstable.

Row-level confirmation shows sparse terminal divergence in the primary paired test: only about 4.19% of the 22,918 seat-runs in the 9000-scenario run had non-zero B-minus-A terminal chip delta. Repeating the same stochastic checkpoint-vs-checkpoint test has diminishing value.

Conclusion: Stage B is materially different from Stage A, but **no reproducible relative strength gain or regression has been established**.

See `docs/LT2_CHECKPOINT_CROSSPLAY_REVIEW_20260917.md`.

## Training decision

Freeze the current SpinCore training line at iteration 7500 / 4.5M roots.

Do not:

- spend another long block merely to increase root count;
- keep rerunning the same A-vs-B cross-play seeking significance;
- change architecture solely because Stage B failed to beat Stage A in these noisy relative tests.

## Immediate next product-strength gate — faithful DeepCrusher R8 v22

The next benchmark must use the faithful DeepCrusher oracle now being built in parallel.

Admission prerequisite:

- literal/structural OpenPPL strategy semantics preserved;
- relevant OpenPPL/library functions implemented faithfully, including true definitions such as `AmountToCall` rather than approximations;
- action sizing/order/priority preserved;
- representative parity probes pass;
- intentional deviations explicitly documented.

Once admitted, benchmark **both Stage A and Stage B** against the exact same DeepCrusher oracle on paired empirical scenarios/deals/seats. Do not assume the later checkpoint is stronger.

See `docs/DEEPCRUSHER_BENCHMARK_CONTRACT_20260917.md`.

## Stop condition

No additional SpinCore training beyond iteration 7500 until the faithful DeepCrusher benchmark is available and reviewed, unless a separately justified bounded training-dynamics experiment is explicitly admitted first.

## Immediate user action

No SpinCore command is required now. Keep the Stage A and Stage B checkpoints intact and continue the DeepCrusher R8 v22 literal C++/OpenPPL-library parity work. Return to this benchmark gate only after the DeepCrusher oracle passes its faithfulness checks.
