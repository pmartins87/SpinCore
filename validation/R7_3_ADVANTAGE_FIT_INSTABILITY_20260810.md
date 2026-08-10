# R7.3 same-memory Advantage fit instability — 2026-08-10

`READY FOR TABLES = NO`. Frozen gates are unchanged.

## Why this control matters

Previous R7.3 work established that CFR-memory variance dominates final AveragePolicy optimizer variance, but it had not cleanly isolated **AdvantageNet fitting variance on one identical Advantage memory**. That distinction matters because each CFR iteration resets AdvantageNet from scratch and immediately sends its predictions through hard regret matching; two similarly accurate regressors can therefore generate very different behavior policies if their predicted regret signs/support differ.

## Physical same-memory experiment

Workflow `31414959700`, evidence commit `d0fa315ee79af013a7e7e3294b0877a0e656f820`, schema `SPINCORE_R7_3_ADVANTAGE_FIT_SIGN_SENSITIVITY_V1`.

One fixed iteration-1 external-sampling reservoir was collected under exact zero-regret uniform behavior:

- 256 roots;
- `10,565` Advantage samples / seen;
- `61,966` traversal nodes.

Four AdvantageNets then trained from scratch on **exactly the same frozen reservoir**. Only initialization and minibatch RNG differed. All four reached the strong internal fit region:

| replica | final weighted NRMSE | optimizer steps |
|---|---:|---:|
| 0 | `0.483531` | 2560 |
| 1 | `0.483770` | 2304 |
| 2 | `0.472173` | 2816 |
| 3 | `0.487427` | 2048 |

Thus every replica is comfortably inside the frozen `0.75` Advantage gate and around the historical-quality `~0.5` internal target.

## Yet the resulting regret policies disagree strongly

Across all six replica pairs on the same 2,048 in-memory observations:

- average hard-regret-matching mean TV: **`0.224349`**;
- average hard-regret-matching p95 TV: **`0.757529`**;
- average fraction of observations with identical positive-regret action support: only **`0.554850`**.

Persisted diagnosis: `ADVANTAGE_FIT_REGRET_SIGN_INSTABILITY_MATERIAL`.

Representative pair `replica_0_vs_1`:

- mean TV `0.225397`;
- p95 `0.739181`;
- max `1.0`;
- legal-action sign disagreement rate `15.51%`;
- identical positive support only `55.76%` of observations.

This is a major causal result: **even if the Advantage memory is held completely fixed, independently fitted networks with NRMSE ~0.48 generate substantially different CFR behavior policies.** Memory/path variance is therefore not the only upstream source of R7.3 divergence.

## Regret-sign fragility

For `replica_0_vs_1`, observations where at least one model has a legal predicted regret very close to zero are disproportionately responsible for policy disagreement:

- normalized min-|regret| <= 0.05: `17.38%` of observations but `22.06%` of total TV mass;
- <= 0.10: `29.49%` of observations and `39.23%` of TV mass;
- <= 0.25: `54.20%` of observations and `70.05%` of TV mass.

Other pairs show the same pattern. Hard positive-regret support is therefore a real amplifier of regression variation near zero.

## Post-hoc smoothing sensitivity

No production rule was changed. As a diagnostic only, a small scale-normalized positive floor was added to legal-action regret weights after prediction.

For `replica_0_vs_1`:

- hard regret matching mean TV `0.225397`, p95 `0.739181`;
- epsilon 0.05 floor: mean `0.203056` (`0.901x`), p95 `0.641930`;
- epsilon 0.10 floor: mean `0.185336` (`0.822x`), p95 `0.574292`.

Across other replica pairs, epsilon 0.10 similarly reduces mean TV to roughly `0.82–0.83x` the hard-rule value. This proves that hard sign/support thresholding contributes materially, but smoothing alone does **not** explain all disagreement: even the post-hoc softened policies remain far apart.

## Immediate engineering consequence

R7.3 now has two distinct upstream variance mechanisms:

1. **target/memory variance** from sampled CFR trajectories;
2. **function-approximation variance**: different AdvantageNet fits to the same memory produce different positive-regret supports and behavior policies.

The next diagnostics were launched immediately rather than serially:

- `31415792326`: exact 2x2 factorial decomposing initialization versus minibatch-order variance on the same frozen memory;
- common Advantage fit-randomness screen: same Advantage initialization plus same per-iteration minibatch RNG across algorithm seeds, while traversal and final AveragePolicy RNG remain seed-specific;
- ongoing five-iteration compounding experiment `31413646505` measures how this behavior divergence evolves after every refit.

If initialization/minibatch randomness is a large component, a deterministic/common Advantage fitting contract may be much cheaper than multiplying tree traversals. If hard-regret sign sensitivity remains dominant after deterministic fitting, the next versioned algorithmic candidates are sign-aware training/calibration or a carefully validated continuous regret mapping—not gate relaxation.
