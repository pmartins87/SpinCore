# R7.3 Advantage estimator design record — 2026-08-10

`READY FOR TABLES = NO`. Frozen R7.3 gates remain unchanged.

## 1. Established causal picture

Physical experiments support:

1. CFR-memory differences dominate AveragePolicy optimizer/init variance;
2. hidden-card stream differences are not dominant;
3. off-support policy extrapolation is material but downstream;
4. stronger Advantage fitting helps but is insufficient;
5. exact controls prove own-reach support fragmentation and opponent external-sampling Advantage noise;
6. at two CFR iterations, four Advantage paths materially improve fitted cross-seed stability;
7. exact opponent expectation is available as a bounded oracle;
8. independent 4 -> 8 path improvement against that oracle is small and the extreme tail remains severe.

The remaining design problem is to find a lower-variance Advantage estimator that remains effective after repeated regret matching and neural refits.

## 2. Short-screen path decomposition

Workflow `31366433008`, evidence `a9c57fe6e3c9149ed3010ead280912295bd4f5f6`:

| mode | mean TV | p95 TV |
|---|---:|---:|
| baseline | `0.305382` | `0.870543` |
| strategy_x4 | `0.275642` | `0.865904` |
| advantage_x4 | `0.219118` | `0.690974` |
| both_x4 | `0.197598` | `0.726534` |

All individual fit gates passed; baseline versus `strategy_x4` kept Advantage checkpoint NRMSE exactly identical. Advantage replication is therefore the strongest isolated short-screen path lever.

## 3. Exact opponent-expectation oracle

Workflow `31368837895`, evidence `45c68d2028ac658ae12870c97b9bf758e47f2a89`:

- exact nodes on four deals/both traversers: `1,265,152`
- exact Advantage samples: `188,440`
- exact phase: `15.91 s`
- max depth: `45`.

Sampled external-sampling memories versus exact expectation:

| paths | exact weight coverage | target relative RMSE | regret-match mean TV | p95 | greedy agreement |
|---:|---:|---:|---:|---:|---:|
| 1 | `0.042562` | `0.835858` | `0.381860` | `1.0` | `0.480239` |
| 4 | `0.110336` | `0.676023` | `0.270441` | `1.0` | `0.733842` |
| 8 | `0.157548` | `0.686965` | `0.257149` | `1.0` | `0.747338` |

Independent x4 captures much of the readily available mean improvement. x8 adds little and leaves p95 `1.0`; x8/x16 is therefore not the preferred next step.

## 4. Replicated-candidate V1: physical results retained, paired interpretation corrected

Three 640 V1 candidates completed and all failed the frozen cross-seed gates:

| V1 candidate | mean TV | p95 TV | individual fits | evidence |
|---|---:|---:|---|---|
| separated Advantage x4 | `0.459596` | `0.898250` | PASS | `94b5e423fa51e1dad8445e6ce36b8832d8161648` |
| separated both x4 | `0.458853` | `0.908883` | PASS | `871967f777f7cec17479ed3ec9f476543452912d` |
| coupled Advantage x4 | `0.451112` | `0.893292` | PASS | `87547311076fd6a015b7d855de1a9c26124b924f` |

A post-run audit found that V1 did not actually use the deterministic deal schedule of the authoritative corrected/strong-Advantage runner.

Authoritative:

```text
deck_seed = seed*1_000_003 + global_root*97 + iteration
```

with `global_root` continuous across iterations.

V1:

```text
deck_seed = (seed<<32) ^ (iteration<<16) ^ root_index_within_iteration
```

with root index restarting each iteration.

Therefore:

- V1 remains evidence that all three configurations are far from the gates on physical independent deal samples;
- V1 must **not** be used for a tightly paired percentage-improvement claim versus corrected 640 or strong-Advantage 640;
- any statement that V1 “matched the acceptance deck schedule” is retracted;
- the exact x4 acceptance delta is being remeasured in V2.

Correction record: `validation/R7_3_REPLICATED_V1_DECK_CONTROL_CORRECTION_20260810.md`.

## 5. Deck-exact coupled x4 V2 — active

`tools/run_r7_3_replicated_640_candidate_v2.py` preserves the V1 collection/fitting machinery but replaces the hidden-deal schedule with the exact generation-2 formula and self-checks continuity at iteration boundaries.

Workflow `31414208511` runs:

- 640 roots/seed;
- four Advantage paths, one strategy path;
- recovered coupled RNG contract;
- exact authoritative deal formula;
- strong Advantage target `0.50`;
- policy target `0.105`, max `32768` steps;
- reservoir `400000`;
- unchanged frozen gates.

This is the authoritative paired x4 acceptance candidate. V1 is not substituted for it.

## 6. Partial-exact V1 — rejected control

Workflow `31369138285` passed build, CTest and 26 Python tests but rejected its physical experiment because an experimental level-0 implementation failed its baseline-control assertion. No positive-level V1 result is accepted.

## 7. Partial-exact V2 — active, `31412806987`

V2 executes level 0 through the authoritative recovered `ExternalSamplingCollector`; only levels 1/2 use probability-weighted exact opponent branching. This removes the previous control ambiguity.

- level 1: enumerate the next opponent decision, then resume external sampling;
- level 2: enumerate the next two opponent decisions.

The smoke passed and the 256-root physical run is active.

## 8. Full-exact upper bound — active, `31412933368`

A bounded experiment compares the authoritative estimator against effectively complete opponent-action expectation (`exact level 128`, beyond observed max depth). It asks whether eliminating opponent path variance entirely produces a large enough stability upper bound to justify further estimator engineering.

If the upper bound is weak, opponent external-sampling variance is real but not sufficient.

## 9. Common-random-number estimator — active, `31413103901`

Modes `independent_1`, `independent_4`, `common_1`, `common_4` test whether correlating opponent-action randomness across training seeds suppresses avoidable iterative amplification.

Under shared decks and iteration-1 uniform behavior, common modes must generate byte-identical Advantage memories across seeds. This is a hard diagnostic invariant.

A production version would require a counter-based RNG contract and checkpoint/resume versioning; the current screen changes no production semantics.

## 10. Five-iteration divergence localization — active, `31413646505`

Parallel baseline and Advantage x4 jobs measure cross-seed **regret-matching behavior** after every Advantage refit, before final AveragePolicy fitting can obscure upstream dynamics.

For each of five iterations the diagnostic records:

- mean/p50/p95/max regret-policy TV;
- per-seed Advantage fit quality;
- strategy-support overlap/target disagreement;
- cumulative roots and memory sizes.

Both jobs passed smoke and are in the physical five-iteration phase. This is the direct test of whether x4 delays divergence for one or two iterations and then loses its advantage at a specific refit.

## 11. Antithetic/rotated-lattice x4 — active, `31413970227`

Ordinary independent x4 is compared with a correlated four-path estimator. For each root/traverser group, all four trajectories share the same underlying Uniform sequence but use offsets `0`, `1/4`, `1/2`, `3/4` modulo 1.

Each individual trajectory is still marginally Uniform and therefore obeys the recovered external-sampling action law. The four-path set is lower-discrepancy/antithetically correlated.

This tests whether the **quality of four samples** can be improved without increasing path count.

## 12. Finite estimator ladder

1. Obtain deck-exact x4 V2 before making a paired acceptance claim about x4.
2. Prefer common-path/counter-based randomness if it materially suppresses iterative divergence at low cost.
3. Prefer bounded partial enumeration if it captures most of the full-exact upper-bound gain efficiently.
4. Prefer antithetic x4 over independent x4 if it improves mean/tail at equal path count.
5. If needed, proceed to stratified opponent-action estimators or control variates benchmarked against the exact oracle.
6. If full exact itself is weak, pivot to regret-sign sensitivity, policy-support discontinuity and target aggregation rather than more opponent sampling.

Not promoted without new causal evidence: more unique roots, independent x8/x16, extra strategy trajectories, card-representation rewrite, or merely more optimizer steps.

Any successful estimator must be explicitly versioned and deterministically recertified for continuous versus stop/restore/continue behavior before R7.3 can close.
