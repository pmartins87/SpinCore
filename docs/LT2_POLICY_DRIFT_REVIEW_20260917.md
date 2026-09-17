# LT2 Policy Drift Review — Stage A 1.8M -> Stage B 4.5M

Date: 2026-09-17
Status: **PASS — MATERIAL POLICY MOVEMENT; WEAK-BASELINE FLATNESS IS NOT EVIDENCE OF STAGNATION**

## Inputs

Stage A finalized checkpoint:

`/home/rz9/spincore_lean_functional/runs/long_training_lt2/20260916_015559/checkpoint.pt`

SHA256:

`e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`

Stage B finalized checkpoint:

`/home/rz9/spincore_lean_functional/runs/long_training_lt2_stage_b/20260917_004911/checkpoint.pt`

SHA256:

`3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`

Diagnostic:

- 3000 empirical SpinGo scenarios;
- 11,040 decision states;
- checkpoint-independent uniform-legal probe trajectories;
- both checkpoints queried on exactly the same solver states;
- total-variation distance, argmax disagreement, entropy and max-probability movement measured overall/by domain/by street.

This is a policy-movement diagnostic only, not a strength/exploitability test.

## Main result

Overall Stage-A -> Stage-B movement is not tiny:

- mean TV: **0.041395**;
- median TV: **0.032119**;
- p95 TV: **0.108574**;
- 91.63% of decisions have TV >= 0.01;
- 26.77% have TV >= 0.05;
- 6.05% have TV >= 0.10;
- argmax action changes on **12.10%** of decisions.

By domain:

- 3H: mean TV 0.038299, p95 0.086771, argmax disagreement 12.46%;
- HU: mean TV 0.047490, p95 0.147864, argmax disagreement 11.40%.

The HU tail is materially heavier than 3H: 12.34% of HU decisions have TV >= 0.10 versus 2.85% in 3H.

## Street concentration

Movement is substantially larger after street 0:

- street 0: mean TV 0.032792, argmax disagreement 9.21%;
- street 1: mean TV 0.072349, argmax disagreement 23.78%;
- street 2: mean TV 0.082074, argmax disagreement 22.75%;
- street 3: mean TV 0.085485, argmax disagreement 15.91%.

HU post-street-0 movement is especially large:

- HU street 1: mean TV 0.094957, p95 0.198585, argmax disagreement 31.08%;
- HU street 2: mean TV 0.115071, p95 0.255831, argmax disagreement 33.96%;
- HU street 3: mean TV 0.122023, p95 0.264341, argmax disagreement 25.00%.

The smaller sample counts on later streets make the street-2/3 estimates noisier, but the concentration is too large to describe the policy as practically unchanged.

## Distribution shape

HU became modestly more mixed on the probe states:

- mean entropy 1.130819 -> 1.144185;
- mean max action probability 0.456671 -> 0.444238.

This is descriptive only; higher entropy is not inherently stronger or weaker.

## Interpretation with the paired weak-baseline review

The Stage A -> Stage B paired weak-baseline review found no statistically distinguishable chip-EV change against uniform-legal, passive-caller or jammer families: all nine checkpoint-delta 95% CIs included zero.

Combined with the policy-drift result, that flatness must **not** be interpreted as evidence that training stopped moving. The AveragePolicy changed materially, particularly in HU post-street-0 decisions, while the weak fixed opponents failed to resolve a corresponding EV difference.

Therefore:

1. do not change architecture merely because the weak-baseline checkpoint delta was flat;
2. do not authorize another long same-regime training block solely because policy movement exists;
3. first measure relative checkpoint strength against a stronger contemporary policy environment and, when ready, the faithful DeepCrusher oracle.

## Immediate next gate — checkpoint cross-play

Run:

`tools/run_lt2_checkpoint_crossplay.sh`

This read-only gate uses 3000 fresh fixed-seed empirical scenarios and evaluates:

- **primary:** Stage B hero minus Stage A hero against the exact same deterministic 50/50 Stage-A/Stage-B opponent mixture, paired by scenario/deal/hero seat/RNG streams;
- **HU direct:** seat-balanced Stage B vs Stage A;
- **3H invasion:** Stage B singleton vs A/A and Stage A singleton vs B/B.

The primary mixture test is the cleanest checkpoint-to-checkpoint comparison because only the hero checkpoint changes while the opponent environment is held fixed.

Cross-play remains a relative checkpoint diagnostic rather than exploitability proof. A faithful DeepCrusher benchmark remains required for product-strength claims.

## Stop condition

Do not resume training beyond iteration 7500 until checkpoint cross-play is reviewed together with policy drift and the weak-baseline evidence.
