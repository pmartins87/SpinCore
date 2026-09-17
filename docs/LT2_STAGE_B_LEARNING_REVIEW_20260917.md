# LT2 Stage B paired learning review — 2026-09-17

## Result

The read-only Stage A -> Stage B comparison completed successfully on 1000 identical fixed-seed empirical scenarios/deals with row-level checkpoint pairing.

Comparison:

- before: LT2 Stage A, 1.8M roots / iteration 3000;
- after: LT2 Stage B, 4.5M roots / iteration 7500;
- opponent families: uniform legal, passive caller, jammer;
- domains: ALL, THREE_HANDED, TRUE_HEADS_UP;
- checkpoint delta paired on scenario, deal, hero seat, opponent family and hero random stream;
- 95% CI clustered by scenario.

## Direct paired checkpoint deltas

| Opponent | Domain | Stage B - Stage A chips/hand | 95% CI |
|---|---|---:|---:|
| Uniform legal | ALL | +0.425 | [-3.592, +4.442] |
| Uniform legal | 3H | -1.020 | [-6.285, +4.245] |
| Uniform legal | HU | +2.087 | [-4.074, +8.248] |
| Passive caller | ALL | -0.795 | [-4.506, +2.917] |
| Passive caller | 3H | -2.115 | [-6.778, +2.548] |
| Passive caller | HU | +0.725 | [-5.187, +6.636] |
| Jammer | ALL | +1.073 | [-2.428, +4.574] |
| Jammer | 3H | +2.974 | [-2.431, +8.380] |
| Jammer | HU | -1.115 | [-5.356, +3.126] |

All nine paired confidence intervals include zero.

## Interpretation

The additional 2.7M roots from Stage A to Stage B did **not produce a statistically distinguishable improvement or regression** under this fixed weak-baseline diagnostic.

The point estimates are mixed rather than directionally coherent:

- ALL: +0.425 / -0.795 / +1.073;
- 3H: -1.020 / -2.115 / +2.974;
- HU: +2.087 / +0.725 / -1.115.

Therefore the earlier reason to defer the HU warning — unsaturated HU AveragePolicy memory — is now exhausted. HU crossed the 2M reservoir threshold at Stage B, yet no measurable weak-baseline gain emerged.

This does **not** prove that the strategy stopped learning in a game-theoretic sense. Weak fixed opponents can become insensitive to policy changes, and this diagnostic is not exploitability/GTO evidence. It does establish that more identical compute is no longer justified solely by the previous weak-baseline learning curve.

## Decision

Do not blindly extend the same training regime beyond 4.5M roots yet.

Before changing architecture or spending another multi-million-root block, distinguish two cases:

1. the AveragePolicy itself barely changed between Stage A and Stage B — evidence of practical policy stagnation/convergence under the current training dynamics;
2. the AveragePolicy changed materially but weak-baseline EV stayed flat — evidence that the weak-baseline diagnostic has become insensitive, in which case a stronger benchmark (faithful DeepCrusher / richer cross-play) is needed before altering training.

## Next finite gate

Run the read-only decision-level policy-drift diagnostic:

`tools/run_lt2_policy_drift_review.sh`

It probes both finalized policies on identical checkpoint-independent uniform-legal solver trajectories and reports total-variation distance, argmax disagreement, entropy and confidence-related summaries overall, by domain and by street.

This is not a strength test. Its purpose is to determine whether Stage A -> Stage B actually moved the learned policy enough to justify blaming the evaluator versus the training dynamics.

Do not continue training beyond iteration 7500 until this gate is interpreted.
