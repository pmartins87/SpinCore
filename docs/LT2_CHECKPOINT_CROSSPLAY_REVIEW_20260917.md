# SpinCore — LT2 Stage A -> Stage B checkpoint cross-play review

Date: 2026-09-17
Status: **TWO INDEPENDENT CROSS-PLAY RUNS COMPLETE — NO REPRODUCIBLE STAGE-B ADVANTAGE OR REGRESSION — EXTERNAL DEEPCRUSHER BENCHMARK NEXT**

## Inputs

Stage A:

- iteration 3000 / 1.8M roots;
- checkpoint SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B:

- iteration 7500 / 4.5M roots;
- checkpoint SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Both evaluations were read-only, used 31 workers, paired scenario/deal/hero-seat/per-seat RNG streams, and changed only the hero checkpoint in the primary deterministic 50/50 Stage-A/Stage-B opponent-mixture test.

## Initial cross-play — 3000 scenarios

Seed `20260918`; 1640 3H / 1360 HU scenarios.

Primary Stage-B-minus-Stage-A result:

| Domain | B - A chips/hand | 95% CI |
|---|---:|---:|
| ALL | -1.8129 | [-3.9564, +0.3306] |
| 3H | -1.9760 | [-4.3207, +0.3687] |
| HU | -1.6162 | [-5.4070, +2.1747] |

Additional diagnostics:

- HU direct B-vs-A: `-0.6165`, CI `[-10.7519,+9.5188]`;
- 3H invasion difference B-vs-AA minus A-vs-BB: `+2.8894`, CI `[-1.2160,+6.9949]`.

The primary point estimates all favored Stage A, but every interval included zero and the 3H invasion diagnostic pointed the opposite way. This was classified borderline/mixed and triggered an independent higher-power confirmation.

## Fresh-seed confirmation — 9000 scenarios

Seed `20260919`; 4918 3H / 4082 HU scenarios.

Primary Stage-B-minus-Stage-A result:

| Domain | B - A chips/hand | 95% CI |
|---|---:|---:|
| ALL | +0.8625 | [-0.4085, +2.1335] |
| 3H | +0.9152 | [-0.5940, +2.4244] |
| HU | +0.7990 | [-1.3337, +2.9317] |

Additional diagnostics:

- HU direct B-vs-A: `+1.5503`, CI `[-4.2649,+7.3656]`;
- 3H invasion difference B-vs-AA minus A-vs-BB: `-2.2202`, CI `[-4.4860,+0.0456]`.

The primary estimates reversed sign relative to the first run and now mildly favor Stage B, but all confidence intervals again include zero. The 3H invasion diagnostic also reversed direction and is nearly negative-significant, while HU direct remains broad.

## Row-level sensitivity observation

The 9000-scenario row evidence explains why checkpoint cross-play is noisy despite material policy drift:

- 22,918 primary mixture hero seat-runs were evaluated;
- only about 4.19% had a non-zero Stage-B-minus-Stage-A terminal chip delta;
- about 10.31% of scenario-cluster means were non-zero;
- 3H seat-run non-zero fraction was about 4.36%;
- HU seat-run non-zero fraction was about 3.90%.

Thus the paired design cancels most trajectories exactly, while a relatively small set of divergent sampled actions produces large positive/negative terminal deltas. This is useful variance reduction, but it also means the remaining strength signal is sparse and unstable across independent seeds.

This observation does not invalidate the evaluator; it explains why simply repeating the same A-vs-B stochastic cross-play has diminishing value.

## Interpretation

The independent 9000-scenario confirmation **did not reproduce** the initial Stage-B disadvantage. It reversed the primary ALL/3H/HU point estimates, yet still failed to establish a statistically distinguishable Stage-B advantage.

Combined with prior evidence:

- weak fixed opponents: no detectable Stage-B improvement/regression;
- policy drift: material decision-distribution movement, especially postflop/HU;
- cross-play run 1: mild/borderline Stage-A direction;
- cross-play run 2: mild Stage-B direction;
- direct/invasion diagnostics: broad and directionally unstable.

Therefore the correct conclusion is **relative checkpoint strength unresolved**, not Stage-B regression and not Stage-B improvement.

The policies are moving materially, but contemporary self-play has not shown that the movement is systematically beneficial.

## Training decision

Pause same-regime SpinCore training at iteration 7500 / 4.5M roots.

Do not chase significance by repeatedly rerunning the same checkpoint-vs-checkpoint cross-play, and do not change architecture from this evidence alone.

The next higher-value gate is the faithful external DeepCrusher R8 v22 benchmark. Benchmark **both Stage A and Stage B** against the same admitted oracle instead of assuming the latest checkpoint is stronger.

See `docs/DEEPCRUSHER_BENCHMARK_CONTRACT_20260917.md`.

## Stop condition

Do not resume SpinCore training beyond iteration 7500 until the faithful DeepCrusher benchmark is available and reviewed, or until a separately justified bounded training-dynamics experiment is explicitly admitted.
