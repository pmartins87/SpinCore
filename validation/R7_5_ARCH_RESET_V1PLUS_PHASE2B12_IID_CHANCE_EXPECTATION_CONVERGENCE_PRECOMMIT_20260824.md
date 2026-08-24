# R7.5 Architecture Reset — Phase2B12 IID Chance-Expectation Convergence

Status: **FROZEN BEFORE PHASE2B12 OUTPUTS**  
Date: 2026-08-24

## 1. Why this diagnostic exists

Phase2B10 established that root target variance is driven materially by **both** opponent-private cards and future public board chance. Phase2B11 then tested an equal-compute crossed private/public estimator and rejected it: FACTOR4X4 worsened pooled mean regret-matching target-policy TV from IID16 `0.33467186760867673` to `0.43390513657870394`, and the direction was worse for both source behavior seeds.

That failure does **not** falsify ordinary IID chance averaging. In the same Phase2B11 readout, IID16 was materially more stable than IID4 (`0.33467` vs `0.52304`). Before changing representation or solver algebra, Phase2B12 measures the convergence curve of a plain conditional IID chance expectation at larger K using one nested sample stream.

The question is deliberately narrow:

> If the acting player's exact initial preflop infoset is held fixed and only legal hidden/future chance is integrated by ordinary IID Monte Carlo, does the raw Advantage target estimate converge far enough, and monotonically enough, to justify a later separately precommitted training pilot?

## 2. Frozen source

- Representation: `H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL`.
- Domain: `THREE_HANDED` only.
- Source behavior trajectories: exact completed Phase2B6.
- Training behavior seeds: `1342191342`, `1801739323`.
- Phase2B6 result SHA-256: `33ec6ba89823dae632b7af935def17444379c96a28e59478c0b7c91f1ec3659a`.
- Phase2B10 result SHA-256: `0295574c6133eb05866ecbdccf7e31efa4e6e8936dbd8bb7e375e166b27fe4dc`.
- Phase2B11 result SHA-256: `1596023d39609ddfe5a6528a2e62d376c8e6bd29dde68d24a20a9b0ed782b1aa`.
- Action candidate: `PF0_CONTROL_33_75_AI`.
- Exact opponent levels: `2`.
- Target iteration: `3`.
- Source behavior contract: Phase2B6 native root, 25% uniform floor on preflop continuation only, postflop native.
- No model fit, optimizer step, reservoir insertion, Strategy collection, AveragePolicy fit, or checkpoint mutation is permitted.

## 3. Frozen geometry

Reuse the exact 15 Phase2B1 root collision groups and the same first 4 anchor deck seeds per scenario used by Phase2B10/B11.

For every `(source behavior seed, scenario, anchor)` create **4 independent IID chance blocks**. Within each block generate a nested stream of **64** legal explicit deals conditional on the acting player's exact two hole cards.

Each deal is generated exactly as the Phase2B11 IID16 control:

- opponent private cards sampled conditionally on actor hole cards;
- public board sampled from the legal remaining deck;
- no hand-strength filtering, bucketing, semantic stratification, rejection sampling, or result-dependent seed choice;
- fixed traversal/action RNG for a scenario/anchor so the measured convergence is chance convergence rather than action-sampling convergence.

The B12 IID seed namespace is deliberately the exact Phase2B11 `IID16` namespace. Therefore samples `0..15` of every block are byte-for-byte the same explicit deals used by B11 IID16. The K16 readout must reproduce the B11 IID16 summaries before B12 output is considered valid.

## 4. Nested estimators

For each block, compute prefix-average raw Advantage target vectors at:

- `K8`  = mean of samples `0..7`;
- `K16` = mean of samples `0..15`;
- `K32` = mean of samples `0..31`;
- `K64` = mean of samples `0..63`.

Raw ten-slot targets are averaged **before** diagnostic regret matching.

The 4 independent blocks are paired deterministically as `(0,1)` and `(2,3)`. For every K and pair report:

- legal target mean absolute difference;
- legal positive/non-positive sign disagreement;
- regret-matching policy total variation;
- dominant legal action mismatch;
- indicator `TV >= 0.35`.

Report pooled and per-source-behavior-seed summaries.

## 5. Exact work

One 64-sample nested stream supplies all four K readouts; smaller K values do not require separate traversals.

Total root target traversals:

`2 behaviors × 15 scenarios × 4 anchors × 4 blocks × 64 = 30,720`.

Workers may be parallelized to 30 independent processes, each with one Torch/OMP/MKL/OpenBLAS thread. Worker count is compute-only and does not change seeds, pairings, chance distribution, or gates.

## 6. Mandatory reproduction gate

Before interpreting K32/K64, B12 K16 must reproduce the exact Phase2B11 IID16 readout within absolute tolerance `1e-12` for:

- pooled mean TV `0.33467186760867673`;
- pooled sign disagreement `0.2520833333333333`;
- pooled target MAD `0.006673636639562426`;
- pooled dominant mismatch `0.3333333333333333`;
- source seed `1342191342` mean TV `0.3171922054847577`;
- source seed `1801739323` mean TV `0.3521515297325957`.

Any failure aborts the diagnostic as `B11_IID16_REPRODUCTION_FAIL` and no scientific classification is accepted.

## 7. Frozen convergence gates

Define the primary candidate as `K64` and the frozen control as reproduced `K16`.

### Material mean-TV improvement

K64 must satisfy **both**:

- absolute mean-TV reduction `>= 0.08`;
- relative mean-TV reduction `>= 25%`.

### Absolute residual target-policy TV

- pooled K64 mean TV must be `<= 0.24`.

This threshold is a screen for a potentially useful solver-level target expectation. It is not the historical final AveragePolicy stability gate.

### Sign stability

K64 sign disagreement must improve from K16 by either:

- `>= 0.05` absolute; or
- `>= 20%` relative.

### Tail stability

Let `tail_rate_035` be the fraction of paired estimator comparisons with regret-matching policy TV `>= 0.35`.

K64 must reduce this rate from K16 by either:

- `>= 0.08` absolute; or
- `>= 20%` relative.

### Seed consistency

- K64 mean TV must be lower than K16 for **both** source behavior seeds.

### Monotone convergence guardrail

- `K32_mean_tv <= K16_mean_tv + 0.01`;
- `K64_mean_tv <= K32_mean_tv + 0.01`.

### Dominant-action guardrail

- K64 dominant-action mismatch rate may not exceed K16 by more than `0.02`.

## 8. Frozen classification and routing

If all gates above pass:

- status: `IID_CHANCE_EXPECTATION_CONVERGES_MATERIALLY`;
- route: `PRECOMMIT_SMALL_MULTI_CHANCE_TARGET_TRAINING_PILOT_WITH_EQUAL_COMPUTE_CONTROL`.

If the full pass fails but K64 improves pooled mean TV by at least `0.05` absolute and both source behavior seeds improve:

- status: `IID_CHANCE_EXPECTATION_CONVERGES_SLOWLY`;
- route: `QUANTIFY_COMPUTE_FRONTIER_OR_REPRESENTATION_SUPPORT_BEFORE_TRAINING`.

Otherwise:

- status: `IID_CHANCE_EXPECTATION_PLATEAUS_OR_UNRESOLVED`;
- route: `REASSESS_REPRESENTATION_AND_REGRET_SIGN_SENSITIVITY_NO_TRAINING`.

A Phase2B12 PASS does **not** authorize production training. It authorizes only a separately precommitted small causal training pilot with an equal-compute control. A SLOW or PLATEAU result authorizes no training pilot.

## 9. Guardrails

- no factorized/crossed estimator retry;
- no Huber beta tuning;
- no lag-anchor tuning;
- no higher continuation uniform floor;
- no result-dependent K choice;
- no seed shopping;
- no threshold relaxation;
- no dropped scenario/anchor/source behavior seed;
- no explicit-deal production training in this phase;
- no optimizer/reservoir/model mutation;
- no architecture winner selection;
- `READY FOR TABLES = NO`.

## 10. Strategic firewall

This diagnostic concerns target-estimator stability only. Even a later small causal training PASS must still satisfy the historical cross-seed stability gates and then a separately precommitted strategic-strength comparison against the certified stable V1 control before architecture selection.
