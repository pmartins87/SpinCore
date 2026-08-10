# R7.3 exact path-variance controls — 2026-08-10

`READY FOR TABLES = NO`. Frozen R7.3 gates are unchanged.

This note records the causal path-variance investigation after the strong-Advantage 640 candidate showed that better neural fit alone improves cross-seed stability only modestly. The exact bootstrap controls remove neural approximation; the downstream four-mode experiment then restores fitted Advantage/AveragePolicy networks to measure which estimator variance actually propagates into the R7.3 policy metric.

## 1. Own-reach AveragePolicy sampling curve

Workflow `31366894171`; evidence commit `3f5130561f0e3d83e65f33b451af86ce80dfa04d`; schema `SPINCORE_R7_3_OWN_REACH_SUPPORT_CURVE_V1`.

The behavior policy is the exact zero-regret uniform policy. No AdvantageNet is fitted. Non-target actions are enumerated as in the recovered own-reach collector; only the target player's own action is sampled. Because the underlying behavior policy is exactly identical, every shared strategy target must be identical. This invariant passed: target TV was zero up to floating-point noise (`<= 2.81e-18`) at every replication level.

| own-reach replicates | poker-isomorphic Jaccard | mean LCFR-weight coverage | shared unique | union unique | shared-target weighted TV |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.033946 | 0.083230 | 1,228 | 36,175 | 0 |
| 2 | 0.041047 | 0.113135 | 2,885 | 70,285 | 0 |
| 4 | 0.052395 | 0.149171 | 6,451 | 123,122 | ~3.65e-19 |
| 8 | 0.074383 | 0.204593 | 18,009 | 242,113 | ~2.81e-18 |

From 1 to 8 trajectories Jaccard rises `2.19120x` and LCFR-weight coverage `2.45818x`, while the true shared targets remain mathematically identical. Persisted diagnosis: `OWN_REACH_SAMPLING_DENSITY_MATERIAL_FOR_SUPPORT`.

**Causal conclusion:** the sampled own-reach collector itself creates major cross-seed support fragmentation even with identical cards and an identical exact policy. This is not a neural-network artifact.

## 2. External-sampling Advantage target curve

Workflow `31366996254`; evidence commit `d1ab3ebc7a905ce2b164e1bf1dee1d5c3efd0a87`; schema `SPINCORE_R7_3_ADVANTAGE_TARGET_CURVE_V1`.

Again behavior is exact uniform and hidden deals are common. No neural fitting occurs. The only seed-dependent quantity is the opponent action sampled during external-sampling Advantage traversal. For states shared by both memories, Advantage targets are aggregated and passed through regret matching so estimator noise is measured in the policy space that can affect the following CFR iteration.

| Advantage path replicates | Jaccard | mean weight coverage | target relative RMSE | regret-matching mean TV | regret-matching p95 TV | weighted greedy agreement |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.028510 | 0.077203 | 1.009432 | 0.421004 | 1.000000 | 0.519830 |
| 2 | 0.027640 | 0.089112 | 0.965445 | 0.407431 | 1.000000 | 0.543558 |
| 4 | 0.033606 | 0.114344 | 0.912991 | 0.382970 | 1.000000 | 0.613636 |
| 8 | 0.048188 | 0.158538 | 0.881045 | 0.371266 | 1.000000 | 0.604422 |

From 1 to 8 trajectories coverage rises `2.05352x`, target relative RMSE falls about `12.7%`, and regret-matching mean TV falls about `11.8%`; p95 remains saturated at `1.0`. Persisted diagnosis: `EXTERNAL_SAMPLING_ADVANTAGE_TARGET_VARIANCE_MATERIAL`.

**Causal conclusion:** Advantage external sampling is a second real path-variance source. Independent replication helps, but the exact bootstrap dose response is gradual rather than transformative.

## 3. Exact own-reach expectation feasibility

Workflow `31367407567`; evidence commit `a16e043fffda04f2c2fa228611e3e352d7ca39b8`; schema `SPINCORE_R7_3_EXACT_OWN_REACH_FEASIBILITY_V1`.

The exact estimator recursively enumerates target-player actions as well as non-target actions. At each target state it records sigma with weight `LCFR_iteration * own_reach`; after target action `a`, reach becomes `own_reach * sigma[a]`. This is the mathematical expectation of the existing sampled own-reach estimator.

The capped four-deal HU benchmark completed:

- 4 unique deals x 2 target players = 8 exact target-root traversals;
- `1,265,152` total visited nodes;
- `188,440` target-state samples;
- `116,192` unique raw observations;
- maximum depth `45`;
- exact traversal phase `4.95 s` on the GitHub CPU runner.

The sampled estimator covered only a tiny subset of exact support on the same deals:

| sampled own-reach trajectories | sampled unique | exact-support coverage | sampled states outside exact support |
|---:|---:|---:|---:|
| 1 | 350 | 0.3012% | 0 |
| 4 | 933 | 0.8030% | 0 |
| 8 | 2,448 | 2.1069% | 0 |

Persisted diagnosis: `EXACT_OWN_REACH_ENUMERATION_FEASIBLE_AT_BENCHMARK_SCALE`.

**Engineering conclusion:** exact own-reach expectation is mathematically attractive but not a drop-in production change. Linear scaling to 640 roots would generate tens of millions of target-state samples, incompatible with the current 100k memory contract. Any use requires a versioned bounded-memory estimator and checkpoint/resume recertification.

## 4. Downstream four-mode path-replication decomposition

Workflow `31366433008`; evidence commit `a9c57fe6e3c9149ed3010ead280912295bd4f5f6`; schema `SPINCORE_R7_3_PATH_REPLICATION_SCREEN_V1`; duration `1201.02 s`.

This experiment restores strong fitted Advantage networks and fitted AveragePolicy networks. It uses the same 256 unique root deals for both algorithm seeds and separates Advantage, strategy and optimizer RNG streams so the two path estimators can be changed independently.

The modes are:

- `baseline`: 1 Advantage path + 1 own-reach strategy path;
- `strategy_x4`: 1 Advantage path + 4 strategy paths;
- `advantage_x4`: 4 Advantage paths + 1 strategy path;
- `both_x4`: 4 Advantage paths + 4 strategy paths.

All four modes passed their individual Advantage and AveragePolicy fit gates. The strict isolation control also passed exactly: the maximum Advantage checkpoint NRMSE difference between `baseline` and `strategy_x4` was **0.0**, so the strategy-only comparison did not contaminate CFR Advantage dynamics.

### Fitted cross-seed policy result

| mode | mean TV | p50 TV | p95 TV | mean ratio vs baseline | p95 ratio vs baseline |
|---|---:|---:|---:|---:|---:|
| baseline | 0.305382 | 0.217856 | 0.870543 | 1.0000 | 1.0000 |
| strategy_x4 | 0.275642 | 0.144226 | 0.865904 | 0.9026 | 0.9947 |
| advantage_x4 | 0.219118 | 0.109186 | 0.690974 | 0.7175 | 0.7937 |
| both_x4 | 0.197598 | 0.069005 | 0.726534 | 0.6471 | 0.8346 |

The cross-mode ratios are meaningful because the experiment controls the unique deals and fitting regime. Absolute values are **not** substituted for the historical or acceptance-run metric because this diagnostic uses a shared-deck/separated-RNG experimental corpus.

`advantage_x4` lowers mean cross-seed TV by about `28.25%` and p95 by about `20.63%`. `strategy_x4` lowers mean only `9.74%` and barely moves p95 (`~0.53%`). `both_x4` gives the lowest mean (`35.30%` below baseline), but its p95 is worse than `advantage_x4` and it costs additional strategy traversals.

On iteration-2 poker-isomorphic strategy support, relative to baseline:

- `advantage_x4`: Jaccard `1.1131x`, LCFR-weight coverage `1.4118x`, shared-target TV `0.8769x`;
- `strategy_x4`: Jaccard `0.9280x`, coverage `1.4033x`, shared-target TV `0.9918x`;
- `both_x4`: Jaccard `0.7594x`, coverage `1.9063x`, shared-target TV `0.9104x`.

Persisted diagnosis: **`ADVANTAGE_EXTERNAL_SAMPLING_VARIANCE_MATERIAL`**.

### Interpretation of the apparent support paradox

The exact uniform-policy control showed that more own-reach paths strongly increase raw strategy support coverage. Yet once CFR behavior is learned, `strategy_x4` alone has only a small effect on fitted cross-seed policy stability, while `advantage_x4` has a much larger effect. These results are consistent rather than contradictory:

1. own-reach sampling determines which strategy observations are stored;
2. but the strategy *being sampled* is itself generated by regret matching on noisy Advantage estimates;
3. reducing Advantage path noise changes the learned behavior policy before strategy collection, so it improves both target agreement and the downstream AveragePolicy;
4. merely collecting the same noisy learned behavior more densely cannot fix the upstream CFR disagreement.

This identifies Advantage external-sampling variance as the first upstream estimator mechanism with a large downstream effect.

## 5. Acceptance-scale promotion

A configurable 640-root replicated candidate runner was smoke-certified by workflow `31368044199` after fixing an experimental-entry metadata import. The full candidate workflow `31368447316` has now launched two matrix jobs in parallel:

### Candidate A — `advantage_x4`

- 640 unique roots/seed;
- 4 independent Advantage external-sampling trajectories per traverser and unique deal;
- 1 own-reach strategy trajectory;
- Advantage fit target `0.50`, max `4096` optimizer steps/iteration;
- AveragePolicy fit target `0.105`, max `32768` optimizer steps;
- reservoir capacity `400000`;
- independent per-algorithm-seed hidden-deal schedule matching acceptance runs.

### Candidate B — `both_x4`

Same acceptance-scale setup, but with 4 own-reach strategy trajectories as well as 4 Advantage trajectories.

Both candidates deliberately use separated Advantage/strategy/optimizer RNG streams because that is the controlled path-replication contract validated by the screen. This is an **experimental production-contract change**, not a silent recovery edit. A gate PASS would therefore be followed by a versioned RNG/replication checkpoint schema and deterministic continuous-vs-stop/restore/continue recertification before R7.3 could close.

The reason for running both now, rather than serially, is that the 256-root screen produced a Pareto tradeoff: `both_x4` has the best mean TV, while `advantage_x4` has the better p95 and lower compute/memory cost. Frozen R7.3 requires both mean and p95, so both are legitimate acceptance-scale candidates.

## 6. Decision after the parallel 640 run

- If `advantage_x4` reaches or materially approaches both frozen cross-seed gates, prefer it over `both_x4` because it is the cleaner isolated estimator correction and cheaper schedule.
- If `both_x4` materially outperforms `advantage_x4` at 640, retain both sources of replication and then optimize the strategy replication factor downward if possible.
- If neither candidate is close enough, do **not** return to brute-force unique-root scaling. The next design step is a lower-variance Advantage estimator (paired/common-random-number, stratified/antithetic opponent-action sampling, or partial enumeration) using the exact controls above as an oracle.
- Exact own-reach expectation remains a secondary versioned estimator option if AveragePolicy support/fit again becomes the limiting factor after upstream Advantage variance is reduced.
