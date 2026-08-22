# R7.5 Architecture Reset — V1+ Phase2B1 Advantage Target Variance Decomposition

Status: **FROZEN BEFORE PHASE2B1 OUTPUTS**  
Date: 2026-08-22

## 1. Purpose

Phase2B0 rejected `RAW_MEAN_THEN_REGRET_MATCH_WITH_CONTROL_EPSILON`: the candidate worsened cross-seed behavior-policy stability and is forbidden from training.

Phase2B1 is a **read-only causal diagnostic**. It does not fit a model, update a reservoir, change a policy, or authorize production. Its purpose is to identify how much instability in H2/THREE_HANDED Advantage targets comes from:

1. stochastic traversal/action sampling with the entire deal held fixed;
2. hidden-card/future-board chance variation while the acting player's exact root SPNNIV3 observation is held fixed;
3. both sources varied together;
4. whether averaging multiple independent target continuations (`K=2/4/8`) would materially suppress sign changes and regret-matching policy divergence before any training ablation is attempted.

## 2. Frozen source state

- Representation: `H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL`.
- Domain: `THREE_HANDED`.
- Source Phase2A execution SHA: `4bfa55d69029cd69536fa6dbfcadd162719cb887`.
- Source Phase2A final behavior ensembles: training seeds `1342191342`, `1801739323`.
- Each source behavior ensemble has four frozen Advantage members.
- Action candidate: `PF0_CONTROL_33_75_AI`.
- Exact opponent levels: `2`.
- Uncertainty algebra/coefficients remain frozen (`epsilon_scale=1.75`, `epsilon_cap=0.5`).
- 15 frozen THREE_HANDED scenarios from `action_scenario_cycle`.

The Phase2B0 evidence file is an input contract and must state `FAIL_DO_NOT_TRAIN_CANDIDATE`.

## 3. Exact-root observation collision design

For every one of the 15 THREE_HANDED scenarios, Phase2B1 deterministically searches deck seeds until it finds one **root infoset bucket with at least 16 distinct deck seeds** producing:

- the exact same authoritative root `SPNNIV3` observation bytes;
- the same acting player;
- the same state-local legal universal action set.

A bucket therefore holds the acting player's observable root state fixed while deck seeds differ. Distinct deck seeds vary hidden opponent cards and/or future board runout conditional on that same observable root infoset.

If any scenario cannot produce a 16-deck collision bucket within the frozen search budget, Phase2B1 fails rather than silently dropping the scenario.

Frozen collision-search budget: `50000` candidate deck seeds per scenario.

## 4. Replicate arms

For each source behavior seed and each of the 15 collision buckets, collect exactly 16 root Advantage target vectors in each arm. Only the root actor is traversed; the root sample is selected by exact observation identity.

### A. `TRAVERSAL_ONLY`

- Same scenario.
- Same exact deck seed for all 16 replicates.
- Same exact root SPNNIV3 observation.
- Different deterministic collector traversal RNG seeds.
- Therefore the full deal/chance realization is fixed; only stochastic opponent-action continuation sampling beyond the two exact opponent levels can differ.

### B. `CHANCE_ONLY`

- Same scenario.
- 16 different deck seeds from the same exact-root-observation collision bucket.
- Same deterministic collector traversal RNG seed for all 16 replicates.
- Therefore the acting player's observable root state and traversal RNG are fixed; hidden/future chance realization differs.

### C. `COMBINED`

- Same 16 deck seeds as `CHANCE_ONLY`.
- Different deterministic traversal RNG seed for each replicate.
- Both chance and stochastic continuation sampling can differ.

No reservoir insertion is permitted. Target samples are captured into an in-memory diagnostic sink only.

## 5. Target identity and extraction

For every replicate:

1. create a fresh solver root from the frozen scenario and requested deck seed;
2. verify root actor and exact root observation match the frozen collision bucket;
3. run `collect_advantage_partial_exact` only for `traverser=root.actor`, with `iteration=3` and `exact_opponent_levels=2`;
4. capture diagnostic Advantage samples in memory;
5. select exactly one sample whose observation equals the exact root observation;
6. require legal mask identity with the frozen bucket;
7. record the ten-slot raw Advantage target vector.

Zero or multiple matching root samples is a hard diagnostic error.

## 6. Frozen K-aggregation screen

For each `(source_behavior_seed, scenario, arm)` there are 16 replicate target vectors. For each `K in {1,2,4,8}`, deterministically form independent left/right aggregates from non-overlapping replicate blocks:

- K=1: 8 left/right comparisons;
- K=2: 4 comparisons;
- K=4: 2 comparisons;
- K=8: 1 comparison.

Within each aggregate, average the **raw Advantage targets slot-wise first**. This is target estimation, not the rejected Phase2B0 behavior-policy algebra change. Each left/right averaged target is then converted to the existing regret-matching policy for diagnostic comparison only.

Record per pair:

- mean absolute target difference on legal slots;
- legal-slot sign-disagreement fraction (`positive` versus `non-positive`, matching regret-matching support semantics);
- regret-matching policy total variation;
- dominant legal action mismatch.

Pool metrics by source behavior seed, arm and K, then across both source behavior seeds.

## 7. Frozen interpretation rules

Phase2B1 may authorize precommitting **one small causal training pilot** only if all applicable criteria below hold.

### K4 materiality

For `COMBINED`, K=4 versus K=1 must:

1. reduce pooled regret-matching policy TV by at least `0.05` absolute **or** `20%` relative;
2. reduce pooled legal-slot sign-disagreement fraction by at least `0.05` absolute **or** `15%` relative;
3. improve regret-matching policy TV directionally for **both** source behavior seeds;
4. show no material curve reversal: `TV(K4) <= TV(K2) + 0.01` and `TV(K8) <= TV(K4) + 0.01`.

If these conditions fail, multi-continuation target averaging is not promoted to training.

### Source classification

Let K=1 pooled policy TV be the source-variance indicator.

- `TRAVERSAL_DOMINANT` if traversal-only K1 TV exceeds chance-only K1 TV by at least 20% relative.
- `CHANCE_DOMINANT` if chance-only K1 TV exceeds traversal-only K1 TV by at least 20% relative.
- otherwise `MIXED_OR_UNRESOLVED`.

This classification is diagnostic and does not itself authorize training.

### Remedy routing

- If `TRAVERSAL_ONLY` materially dominates and K4 passes there, the next candidate is a same-deal multi-traversal continuation estimator.
- If `CHANCE_ONLY` materially dominates and K4 passes there, the next work must target conditional chance/deck variance or stratified/common-random-number chance support; simply repeating action sampling on the same deal is insufficient.
- If `COMBINED` K4 fails materiality, do not spend a training run on generic K4 continuation averaging.

## 8. Compute contract

- Diagnostic only; no optimizer step and no model fit.
- Fresh solver traversal is allowed because the experiment measures target-generation variance.
- Ryzen launcher uses process isolation for independent scenario/arm tasks.
- Frozen launcher worker count: `12` processes maximum.
- Each worker uses one Torch/OMP/MKL thread to avoid oversubscription.
- Statistical outputs must be invariant to worker completion order; all aggregation is sorted by frozen task identity before metrics are computed.

## 9. Governance

- R7.5.3 remains `FAIL_BLOCKED_CLOSED`.
- No H2/H3 representation winner exists.
- R7.5.4 and R8 remain blocked.
- `READY FOR TABLES = NO`.
- Phase2B1 cannot authorize production training.
- A Phase2B1 PASS only permits freezing one small causal pilot whose strategic strength remains a separate mandatory gate later.
