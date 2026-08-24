# R7.5 Architecture Reset — Phase2B11 Factorized Private/Public Chance Estimator Screen

Status: **FROZEN BEFORE PHASE2B11 OUTPUTS**  
Date: 2026-08-24

## 1. Why this screen exists

Phase2B10 separated the chance source at the exact initial THREE_HANDED preflop root using the explicit-deal diagnostic solver extension. Both components are materially active:

- traversal-only K1 policy TV: `0.05516957953078942`;
- opponent-private-only: `0.40876631080266784`;
- future-public-board-only: `0.4470753497550347`;
- combined private+public: `0.5439893672482926`.

Neither private nor public chance dominates consistently across both frozen behavior seeds. The combined arm also remains about `0.0969` TV above the stronger pooled single-component arm. Therefore the next question is not which one component to remove, but whether a **factorized Monte-Carlo estimator** can reduce the variance created by both components at equal traversal budget.

Phase2B1 already showed that naive raw-target averaging across ordinary combined chance samples is not automatically beneficial. Phase2B11 therefore compares a crossed private/public design directly against ordinary IID combined sampling at the **same number of root traversals**. No training is authorized by this precommit.

## 2. Frozen source and prerequisites

- Representation: `H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL`.
- Domain: `THREE_HANDED` only.
- Frozen behavior: exact completed Phase2B6 with 25% uniform floor on preflop continuations only.
- Behavior seeds: `1342191342`, `1801739323`.
- Phase2B6 result SHA-256: `33ec6ba89823dae632b7af935def17444379c96a28e59478c0b7c91f1ec3659a`.
- Phase2B10 result SHA-256: `0295574c6133eb05866ecbdccf7e31efa4e6e8936dbd8bb7e375e166b27fe4dc`.
- Phase2B10 classification: `MIXED_PRIVATE_PUBLIC_CHANCE`.
- Action candidate: `PF0_CONTROL_33_75_AI`.
- Exact opponent levels: `2`.
- Target iteration: `3`.
- Same 15 Phase2B1 exact-root collision groups and first 4 stored anchor deals per scenario.
- Root actor hole cards, root SPNNIV3 bytes, actor and universal legal mask must remain exact for every generated deal.

## 3. Sampling construction

The acting player's two root hole cards are fixed by the anchor. For each estimator block, chance is generated from two independent deterministic seed namespaces:

1. a **private seed** generates an ordered four-card opponent-hole assignment from the 50 cards not held by the actor;
2. a **public seed** generates a random ordering of the same 50-card actor-excluded deck. After removing the four selected opponent cards, the first five remaining cards form the board.

For any fixed private assignment, filtering a uniformly random 50-card ordering after removing those four cards induces a uniformly random ordered five-card board from the valid 46-card remainder. Thus every individual crossed cell has the correct conditional deal distribution. Sharing a public random ordering across different private rows is a variance-reduction coupling only; it does not change any cell's marginal legal deal distribution.

All generated deals are checked for card range, uniqueness, fixed actor hole cards, exact root SPNNIV3 observation, actor and legal identity before target collection.

## 4. Estimator arms and equal-compute controls

Each `(behavior seed, scenario, anchor, block)` produces four estimators:

### A. `IID4`

Four independent `(private seed, public seed)` pairs. Average the four raw ten-slot Advantage target vectors, then apply diagnostic regret matching once.

### B. `FACTOR2X2`

Two independent private rows × two independent shared public columns = four crossed valid deals. Average all four raw target vectors, then apply diagnostic regret matching once.

This uses exactly the same four root traversals as `IID4`.

### C. `IID16`

Sixteen independent `(private seed, public seed)` pairs. Average the sixteen raw target vectors, then apply diagnostic regret matching once.

### D. `FACTOR4X4`

Four independent private rows × four independent shared public columns = sixteen crossed valid deals. Average all sixteen raw target vectors, then apply diagnostic regret matching once.

This uses exactly the same sixteen root traversals as `IID16`.

There is no weighting by observed result, hand strength, action or target. All means are simple arithmetic means of raw target vectors before regret matching.

## 5. Independent estimator blocks

Use **4 deterministic independent blocks** per `(behavior seed, scenario, anchor, arm)`. The same fixed traversal RNG is used across all chance draws and all four blocks for that scenario/anchor, so block-to-block disagreement is driven by the chance estimator rather than traversal RNG.

Estimator blocks are paired non-overlapping as `(0,1)` and `(2,3)`. For each pair compute:

- legal target mean-absolute difference;
- legal positive/non-positive sign disagreement;
- regret-matching policy TV;
- dominant legal action mismatch.

The scientific comparison is factorized versus IID at the same traversal budget.

## 6. Exact work

Per `(behavior seed, scenario, anchor, block)`:

`IID4 4 + FACTOR2X2 4 + IID16 16 + FACTOR4X4 16 = 40` root target traversals.

Total:

`2 behavior seeds × 15 scenarios × 4 anchors × 4 blocks × 40 = 19,200` root target traversals.

Workers may be parallelized at the independent scenario/anchor/block task level. Each worker must use one Torch/OMP/MKL/OpenBLAS thread. Worker count is compute-only.

## 7. Primary frozen gate

`FACTOR4X4` is the primary screen against equal-compute `IID16`.

A factorized chance estimator is considered supported only if **all** of the following hold:

1. pooled mean RM-policy TV improvement `IID16 - FACTOR4X4` is at least `0.05` absolute **or** at least `20%` relative;
2. pooled legal sign-disagreement improvement is at least `0.05` absolute **or** at least `15%` relative;
3. `FACTOR4X4` mean RM-policy TV is lower than `IID16` separately for both source behavior seeds;
4. pooled `FACTOR4X4` p95 RM-policy TV is no more than `IID16 p95 + 0.02`;
5. pooled `FACTOR4X4` dominant-action mismatch rate is no more than `IID16 + 0.02`;
6. no factor-size reversal: pooled `FACTOR4X4` mean RM-policy TV must be no more than pooled `FACTOR2X2 + 0.01`.

These thresholds are frozen before Phase2B11 output inspection.

## 8. Secondary compute-efficiency readout

`FACTOR2X2` versus `IID4` is diagnostic. Report the same absolute/relative improvements and by-seed directions. A 2×2 result cannot override a failed primary 4×4 gate post hoc.

If the primary 4×4 gate passes, Phase2B11 may report whether 2×2 appears promising enough to consider in the later pilot design, but it does not automatically select a training configuration.

## 9. Frozen routing

- Primary PASS -> `PRECOMMIT_SMALL_FACTORIZED_CHANCE_TARGET_TRAINING_PILOT_WITH_EQUAL_COMPUTE_CONTROL`.
- Primary FAIL but clear raw-target/sign improvement without policy-TV improvement -> `INVESTIGATE_REGRET_MATCHING_SENSITIVITY_AFTER_FACTORIZED_TARGET_ESTIMATION_NO_TRAINING`.
- Primary FAIL without coherent estimator improvement -> `REASSESS_SOLVER_LEVEL_CHANCE_EXPECTATION_OR_REPRESENTATION_SUPPORT_NO_TRAINING`.

No Phase2B11 result authorizes production training or table use. Even a primary PASS authorizes only preparation of one separately precommitted small causal trajectory pilot.

## 10. Guardrails

- no Huber beta tuning;
- no lag-anchor tuning;
- no higher uniform floor;
- no seed shopping;
- no threshold relaxation;
- no dropped scenario, anchor, block or source behavior seed;
- no result-dependent resampling;
- no AveragePolicy fit;
- no optimizer step;
- no reservoir insertion;
- no checkpoint mutation;
- no architecture winner selection;
- `READY FOR TABLES = NO`.

## 11. Strategic firewall

Factorized chance sampling is a stability mechanism candidate only. If a later causal training pilot passes the historical stability gates, strategic strength must still be evaluated separately against the certified stable V1 control before any architecture selection.
