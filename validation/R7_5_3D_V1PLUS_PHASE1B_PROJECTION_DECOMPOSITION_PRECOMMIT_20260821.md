# R7.5.3D — V1+ Phase 1B projection decomposition precommit

Date: 2026-08-21
Status: FROZEN_BEFORE_PHASE1B_OUTPUTS
READY FOR TABLES: NO
Production training authorized: NO

## Purpose

Resolve the remaining identification gap after the Phase-1 forensic post-mortem without training any model.

Phase 1 showed that 3H Strategy memory is under radically greater pressure than HU and that 3H instability already exists at `action_path_len=0`, so full voluntary-history richness cannot be the sole/root cause. However, the prior reservoir diagnostic's exact-observation overlap mixes cards, current chip geometry, legality and history. Phase 1B separates those components.

## Frozen inputs

- Existing final x16 checkpoints only.
- Training execution SHA: `f44e05513721b59f63ed5c61f37de2c115c67315`.
- Representations: H2 and H3 final.
- Domains: TRUE_HEADS_UP and THREE_HANDED.
- Training seeds: 1342191342 and 1801739323.
- Both final Advantage and Strategy reservoirs.

No new roots, traversal, optimizer step, reservoir mutation, threshold change, seed change, or model-weight change is allowed.

## Frozen projections

For every retained reservoir observation, compute deterministic identities for:

1. `exact_observation`: authoritative complete SPNNIV3 observation bytes.
2. `cards_only`: canonical rank tokens plus same-suit relations.
3. `geometry_only`: current categorical state, current numeric chip geometry and primitive legality, excluding cards and public history.
4. `fixed_state_no_history`: cards + current categorical/numeric geometry + legality, excluding public history.
5. `history_exact`: complete event categorical and quantitative history.
6. `history_structured`: complete actor/street/action/forced categorical history, excluding quantitative event fields.
7. `history_v1_like`: last 32 `(street, action_type)` events.
8. `no_cards_plus_exact_history`: current geometry/legality + exact history.
9. `no_cards_plus_structured_history`: current geometry/legality + structured categorical history.
10. `no_cards_plus_v1_history`: current geometry/legality + V1-like history.

For each projection, record unique counts per seed and cross-seed intersection/union/Jaccard.

## Decision logic frozen before outputs

- If `history_exact` overlap is materially lower than `history_structured`/`history_v1_like` in 3H Strategy memory while card/current-state projections are comparatively well aligned, quantitative history compression becomes the first causal ablation.
- If `history_exact`, `history_structured`, and `history_v1_like` are similarly aligned but `cards_only` or `fixed_state_no_history` overlap collapses in 3H, the dominant problem is chance/current-state support rather than event-history numeric detail; prioritize sampling/variance/memory treatment.
- If history-only overlaps are high but `geometry_only` is poor in 3H, prioritize state-space/sampling stratification around stack/pot/position geometry.
- If all coarse projections are reasonably aligned while exact/full state overlap remains poor, model capacity/regularization becomes a stronger candidate, but only after ruling out reservoir support imbalance.
- H3 semantics are not changed or selected by this readout.
- No representation winner may be declared from Phase 1B alone.

## Guardrails

- Correlation/overlap is diagnostic, not a strategic-strength metric.
- No production training is authorized.
- No stability threshold is relaxed.
- No domain may be dropped.
- The next training experiment, if any, must be a small causal ablation frozen only after Phase 1B output is persisted.
