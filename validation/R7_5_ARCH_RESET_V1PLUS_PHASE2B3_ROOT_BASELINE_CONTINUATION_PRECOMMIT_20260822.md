# R7.5 Architecture Reset — V1+ Phase2B3 Root-Baseline / Downstream-Continuation Decomposition

Status: **FROZEN BEFORE PHASE2B3 OUTPUTS**  
Date: 2026-08-22

## 1. Purpose

Phase2B2 returned `MIXED_CHANCE_SUPPORT_AND_FEEDBACK`. Evaluating the two already-diverged H2/THREE_HANDED behavior ensembles on identical stored chance realizations reduced K1 cross-behavior regret-matching-policy TV from `0.5153716032136447` to `0.38892191351328625`, but failed the frozen shared-support gate.

Phase2B3 is a read-only causal decomposition of the **remaining same-chance behavior feedback**. It asks whether that residual divergence is driven primarily by:

1. the root traverser's own current behavior policy `sigma`, which defines the CFR node-value baseline subtracted from all root action values; or
2. different downstream continuation values produced by the two already-diverged behavior ensembles on the same deal; or
3. both / nonlinear interaction.

No training, optimizer step, reservoir insertion, checkpoint mutation, or architecture selection is permitted.

## 2. Source identity

- Representation: `H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL`.
- Domain: `THREE_HANDED`.
- Action candidate: `PF0_CONTROL_33_75_AI`.
- Exact opponent levels: `2`.
- Target iteration tag: `3`.
- Source Phase2A execution SHA: `4bfa55d69029cd69536fa6dbfcadd162719cb887`.
- Source behavior seeds: `1342191342`, `1801739323`.
- Phase2B1 result SHA-256: `f95751afeb17fcd5844bfcb2971577b92a400750444e5dabe2f4ddb5718ba6ef`.
- Phase2B2 result SHA-256: `49cd1bd98ffe30f21a2b4263c50eb0b5c6d3e616b651a1353f136a670453e281`.
- Frozen Phase2B2 primary K1 independent-RNG TV to reproduce: `0.38892191351328625`.

Use exactly the 15 Phase2B1 collision groups and their 16 stored deck seeds. No new deck search is allowed.

## 3. Exact root-component reconstruction

For each stored deal and each source behavior ensemble, reconstruct the Phase2B2 `INDEPENDENT_TRAVERSAL_RNG` root target exactly.

At the root traverser node:

- evaluate the source behavior policy `sigma` on the exact root observation;
- enumerate every legal root action in canonical legal-slot order;
- recursively evaluate each child with the same collector semantics, exact-opponent level `2`, and the exact Phase2B2 independent traversal-RNG namespace for that source behavior;
- preserve RNG evolution across root actions exactly as the authoritative collector does;
- obtain the legal root action-value vector `V`;
- reconstruct `node_value = dot(sigma, V)` and `target = V - node_value` on legal slots.

The pooled native target-policy TV must reproduce Phase2B2 K1 independent TV within `1e-12`, otherwise Phase2B3 is invalid.

## 4. Frozen counterfactuals

For every paired deal, let `(V_A, sigma_A)` and `(V_B, sigma_B)` be the two source-behavior root components.

### A. `NATIVE`

Compare:

- `T(V_A, sigma_A)` versus
- `T(V_B, sigma_B)`.

This must reproduce the Phase2B2 primary result.

### B. `COMMON_ROOT_SIGMA`

Define `sigma_bar = 0.5 * (sigma_A + sigma_B)` on legal actions (renormalized defensively). Compare:

- `T(V_A, sigma_bar)` versus
- `T(V_B, sigma_bar)`.

This removes **root-baseline-policy disagreement** while preserving each source behavior's downstream continuation values. A large reduction from NATIVE means root baseline feedback materially amplifies instability.

### C. `COMMON_ACTION_VALUES`

Define `V_bar = 0.5 * (V_A + V_B)` slot-wise. Compare:

- `T(V_bar, sigma_A)` versus
- `T(V_bar, sigma_B)`.

This removes **downstream continuation-value disagreement** while preserving the two native root policies. A large reduction from NATIVE means downstream behavior feedback materially amplifies instability.

### D. Crossed path diagnostics

Also compute the four target constructions `T(V_A,sigma_A)`, `T(V_A,sigma_B)`, `T(V_B,sigma_A)`, `T(V_B,sigma_B)` and report symmetric path-step magnitudes:

- root-sigma step magnitude = mean of `TV[T(V_A,sigma_A),T(V_A,sigma_B)]` and `TV[T(V_B,sigma_A),T(V_B,sigma_B)]`;
- downstream-value step magnitude = mean of `TV[T(V_A,sigma_A),T(V_B,sigma_A)]` and `TV[T(V_A,sigma_B),T(V_B,sigma_B)]`.

These are diagnostics only and are not assumed additive because regret matching is nonlinear.

## 5. Metrics

Pool across all `15 * 16 = 240` paired deals and also report per scenario:

- root behavior-policy TV `TV(sigma_A, sigma_B)`;
- legal-slot root action-value mean absolute difference;
- NATIVE regret-matching-policy TV and sign disagreement;
- COMMON_ROOT_SIGMA regret-matching-policy TV and sign disagreement;
- COMMON_ACTION_VALUES regret-matching-policy TV and sign disagreement;
- crossed root-sigma and downstream-value path-step TV summaries.

## 6. Frozen materiality and routing

Let `native_tv` be the pooled NATIVE mean TV, `common_sigma_tv` the pooled COMMON_ROOT_SIGMA mean TV, and `common_values_tv` the pooled COMMON_ACTION_VALUES mean TV.

A factor-removal effect is **material** if removing that factor reduces NATIVE TV by at least `0.05` absolute **or** `15%` relative.

- `ROOT_BASELINE_DOMINANT` if root-sigma removal is material and downstream-value removal is not.
- `DOWNSTREAM_CONTINUATION_DOMINANT` if downstream-value removal is material and root-sigma removal is not.
- `MIXED_ROOT_AND_DOWNSTREAM_FEEDBACK` if both removals are material.
- `NONLINEAR_INTERACTION_OR_UNRESOLVED` if neither is material.

No Phase2B3 classification authorizes training. Routing is diagnostic only:

- root-baseline dominant -> freeze a read-only/very-small screen of baseline-policy stabilization mechanisms before training;
- downstream-continuation dominant -> localize feedback by street/depth before training;
- mixed -> localize downstream street/depth and separately retain root-baseline stabilization as a required causal control;
- unresolved -> inspect interaction by action/scenario before training.

## 7. Compute contract

- Exactly 15 scenarios × 16 stored deals × 2 source behaviors = 480 root-component reconstructions.
- Up to 12 worker processes.
- One Torch/OMP/MKL thread per worker.
- Each worker loads both frozen four-member source behavior ensembles once.
- Diagnostic sinks may capture recursive samples in memory but must never write a training reservoir.
- Aggregation is deterministic after sorting by `(scenario_index, replicate)`.

## 8. Governance

- R7.5.3 remains `FAIL_BLOCKED_CLOSED`.
- Phase2B0 candidate remains forbidden from training.
- Phase2B1 generic K4 remains rejected.
- Phase2B2 shared-support gate remains failed.
- `READY FOR TABLES = NO`.
- `production_training_authorized = false`.
