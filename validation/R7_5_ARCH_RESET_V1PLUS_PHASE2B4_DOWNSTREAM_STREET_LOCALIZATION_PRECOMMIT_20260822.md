# R7.5 Architecture Reset — V1+ Phase2B4 Downstream Street-Depth Localization

Status: **FROZEN BEFORE PHASE2B4 OUTPUTS**  
Date: 2026-08-22

## 1. Purpose

Phase2B3 reproduced the Phase2B2 same-chance native K1 target-policy TV exactly at `0.38892191351328625` and decomposed the residual feedback into two material components:

- common root sigma residual TV `0.32770276958712846` (root-baseline effect ≈15.7% relative);
- common action values residual TV `0.21015032820513388` (downstream-continuation effect ≈46.0% relative).

The next unresolved question is **where in downstream street depth the continuation-policy divergence enters**. Phase2B4 is a read-only causal diagnostic. It keeps the root CFR baseline common between the two source behaviors and progressively replaces downstream behavior policies with their pointwise arithmetic-mean policy from a frozen street onward.

No training, model fit, optimizer step, reservoir insertion, checkpoint mutation, representation selection, or production authorization is permitted.

## 2. Frozen source identity

- Representation: `H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL`.
- Domain: `THREE_HANDED`.
- Action candidate: `PF0_CONTROL_33_75_AI`.
- Exact opponent levels: `2`.
- Target iteration tag: `3`.
- Source Phase2A execution SHA: `4bfa55d69029cd69536fa6dbfcadd162719cb887`.
- Source behavior seeds: `1342191342`, `1801739323`.
- Phase2B1 result SHA-256: `f95751afeb17fcd5844bfcb2971577b92a400750444e5dabe2f4ddb5718ba6ef`.
- Phase2B2 result SHA-256: `49cd1bd98ffe30f21a2b4263c50eb0b5c6d3e616b651a1353f136a670453e281`.
- Phase2B3 result SHA-256: `158e450e96027871b5bf44caa5cd0cb9105782e648e91583960b49d3986fe0a7`.
- Frozen Phase2B3 common-root-sigma TV to reproduce: `0.32770276958712846`.

Use exactly the 15 Phase2B1 collision groups and exactly their 16 stored deck seeds per scenario. No new collision search or chance-support selection is allowed.

## 3. Root-baseline control

For every stored deal, evaluate the two frozen source behavior ensembles at the exact root observation and compute:

`SIGMA_BAR = normalize(0.5 * (SIGMA_A + SIGMA_B))`

Every Phase2B4 target uses this same `SIGMA_BAR` when centering its root action-value vector. Therefore the root-sigma amplification identified by Phase2B3 is controlled out in all Phase2B4 arms.

The `NATIVE_CONTINUATION` arm must reproduce Phase2B3 `COMMON_ROOT_SIGMA` pooled TV within `1e-12` before any street-depth interpretation is accepted.

## 4. Nested downstream continuation arms

Street IDs use the authoritative solver metadata: preflop `0`, flop `1`, turn `2`, river `3`.

For each source side A/B, define a hybrid policy provider. At states earlier than the arm threshold it uses that source side's native frozen behavior policy. At states at or after the threshold, **both sides use the exact same pointwise mean behavior policy**:

`MEAN_POLICY(state) = normalize(0.5 * (POLICY_A(state) + POLICY_B(state)))`.

Frozen arms:

1. `NATIVE_CONTINUATION`: no downstream policy commonization.
2. `COMMON_FROM_RIVER`: common policy only on river states.
3. `COMMON_FROM_TURN`: common policy on turn and river states.
4. `COMMON_FROM_FLOP`: common policy on flop, turn and river states.
5. `COMMON_FROM_PREFLOP`: common policy on every downstream state after the chosen root action, including remaining preflop decisions.

This nested design estimates cumulative feedback removed by progressively commonizing deeper-to-earlier streets. Sequential differences are reported as localization diagnostics but are **not assumed additive** because traversal path selection, target centering and regret matching are nonlinear.

## 5. Traversal RNG and chance support

For every scenario/deal/source side, use the exact Phase2B3 independent traversal RNG namespaces:

- behavior A namespace `1`;
- behavior B namespace `2`.

Each arm restarts from the same deterministic RNG seed for that `(scenario, replicate, source side)` so arm differences are due to policy commonization rather than different RNG streams.

Chance support is identical between source sides because each pair uses the same stored deck seed.

## 6. Root action-value reconstruction

For every arm/source side/deal:

1. reconstruct the exact root;
2. verify exact SPNNIV3 observation SHA, root actor, legal action set and scenario identity;
3. for each root legal action, create the exact child and evaluate the downstream utility using the frozen audited partial-exact recursion with `exact_opponent_levels=2`;
4. collect the ten-slot root action-value vector;
5. center both source vectors using the same frozen `SIGMA_BAR` for that deal;
6. compare target vectors using mean absolute difference, sign disagreement, regret-matching policy TV and dominant legal-action mismatch.

No Advantage/Strategy sample may be inserted into any training reservoir.

## 7. Frozen summaries and localization quantities

For each arm, pool all 240 paired deals and record the same target metrics used by Phase2B3.

Let `T_NATIVE` be `NATIVE_CONTINUATION` pooled target-policy TV. Let `T_RIVER`, `T_TURN`, `T_FLOP`, `T_PREFLOP` be the corresponding commonization-arm TVs.

Report cumulative reductions versus native:

- river-only: `T_NATIVE - T_RIVER`;
- turn+river: `T_NATIVE - T_TURN`;
- flop+turn+river: `T_NATIVE - T_FLOP`;
- all downstream: `T_NATIVE - T_PREFLOP`.

Also report sequential nested steps:

- river step: `T_NATIVE - T_RIVER`;
- turn step: `T_RIVER - T_TURN`;
- flop step: `T_TURN - T_FLOP`;
- remaining-preflop step: `T_FLOP - T_PREFLOP`.

Negative steps are retained and never clipped.

## 8. Frozen interpretation rules

A cumulative effect is **material** if it reduces pooled target-policy TV by at least `0.05` absolute or `15%` relative to `T_NATIVE`.

Classification:

- `POSTFLOP_FEEDBACK_DOMINANT` if `COMMON_FROM_FLOP` is material and the additional `COMMON_FROM_PREFLOP` reduction is not material.
- `PREFLOP_AND_POSTFLOP_FEEDBACK_MIXED` if `COMMON_FROM_FLOP` is material and the additional `COMMON_FROM_PREFLOP` reduction is also material.
- `PREFLOP_DOWNSTREAM_FEEDBACK_DOMINANT` if `COMMON_FROM_FLOP` is not material but `COMMON_FROM_PREFLOP` is material.
- `DEPTH_LOCALIZATION_WEAK_OR_UNRESOLVED` if neither cumulative cutoff is material.

Independent residual guardrail: if `COMMON_FROM_PREFLOP` pooled TV remains above `0.10`, append `HIGH_RESIDUAL_AFTER_FULL_POLICY_COMMONIZATION=true`; this blocks any immediate training candidate and requires diagnosing residual stochastic/value-estimation effects first.

Phase2B4 does not itself authorize training. Its only allowed output is a localization route for the next precommit.

## 9. Compute contract

- 15 scenarios × 16 stored deals × 2 source sides × 5 arms = `2400` root action-value reconstructions.
- Up to 12 independent worker processes.
- One Torch/OMP/MKL thread per worker.
- Each worker loads both frozen four-member source behavior ensembles once.
- All aggregation is sorted by `(scenario_index, arm)` and independent of worker completion order.

## 10. Governance

- R7.5.3 remains `FAIL_BLOCKED_CLOSED`.
- No H2/H3 winner exists.
- R7.5.4 and R8 remain blocked.
- `READY FOR TABLES = NO`.
- Phase2B4 cannot authorize production training.
