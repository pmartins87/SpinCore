# R7.5 Architecture Reset — V1+ Phase2B7 Residual Localization

Status: **FROZEN BEFORE PHASE2B7 OUTPUTS**  
Date: 2026-08-23

## 1. Purpose

Phase2B6 established a real causal effect from the 25% preflop-continuation training floor, but the learned AveragePolicy remains outside the historical cross-seed stability gates. The COMMON pooled mean TV improved from `0.21376380367532596` to `0.18934816676149685`, with a paired 95% bootstrap CI for baseline-minus-pilot of `[0.015543579793138763, 0.03274093539153998]`, yet the two heldouts remain at mean TV `0.18810851478911766` / `0.19058781873387604` and p95 `0.48880605139812816` / `0.5286989995226848`.

Phase2B7 is therefore a **read-only residual-localization diagnostic**. It asks where the remaining cross-seed disagreement lives after the successful causal intervention. It does not alter training, policies, reservoirs, heldout states, action abstraction, or thresholds.

The diagnostic must not be used to justify a larger 50%, 75%, or 100% damping floor. Those remain prohibited.

## 2. Frozen inputs

- Phase2B6 execution SHA: `4fa96434321c32efc734a55ae75982018ff2d091`.
- Exact Phase2B6 result SHA-256: `33ec6ba89823dae632b7af935def17444379c96a28e59478c0b7c91f1ec3659a`.
- Exact Phase2A result SHA-256: `65f691e6b9cf7fbbddf88852c5ac6e0dcd2211af45f53cc4bb3e8271dbaa6149`.
- Representation: `H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL`.
- Domain: `THREE_HANDED`.
- Training seeds: `1342191342`, `1801739323`.
- Heldout evaluation seeds: `2029384436`, `1150634112`.
- First `1024` frozen states per heldout, identical to Phase2A/Phase2B6 evaluation.
- Primary readout: `COMMON_LEARNER`.
- Corroborative readout: `NATIVE_LEARNER`.
- Phase2A control arm: exact `S100K_CONTROL` policies.
- Phase2B6 pilot policies: exact completed `COMMON_LEARNER` / `NATIVE_LEARNER` artifacts from the Phase2B6 output directory.

Before analysis, the tool must reproduce the Phase2B6 heldout mean and p95 metrics to numerical tolerance from the exact local artifacts. A mismatch invalidates the diagnostic.

## 3. Frozen state partitions

Each heldout state is classified directly from authoritative SPNNIV3 bytes and the frozen descriptor metadata.

Primary causal regions:

1. `PREFLOP_ROOT`: street=preflop and zero non-forced preflop public events.
2. `PREFLOP_CONTINUATION_1`: street=preflop and exactly one non-forced preflop public event.
3. `PREFLOP_CONTINUATION_2PLUS`: street=preflop and at least two non-forced preflop public events.
4. `FLOP`.
5. `TURN`.
6. `RIVER`.

Secondary partitions are frozen before output inspection:

- actor: seat `0`, `1`, `2`;
- scenario index: `0..14`;
- action-path length bins: `0`, `1`, `2`, `3`, `4-5`, `6+`;
- legal-action count: exact count;
- total SPNNIV3 history-count bins: `0`, `1`, `2`, `3`, `4-5`, `6-9`, `10+`.

No partition may be added after seeing the Phase2B7 output and then presented as precommitted evidence.

## 4. Metrics

For every partition and for both learner modes report:

- state count;
- baseline Phase2A cross-seed TV mean, p50, p95 and max;
- pilot Phase2B6 cross-seed TV mean, p50, p95 and max;
- absolute and relative mean improvement;
- share of total pilot TV mass (`sum(group TV) / sum(all TV)`);
- count and share of states with pilot TV `> 0.35`;
- share of all `TV > 0.35` tail states contributed by the partition.

Also report the top 50 COMMON pilot residual states by TV with evaluation seed, state index, scenario, actor, street, path length, non-forced preflop count, history count, legal slots, baseline TV, pilot TV and baseline-minus-pilot delta.

The historical `0.35` value is used here only to localize the tail that blocks the p95 stability gate; it is not a new threshold.

## 5. Frozen routing rule

The routing rule uses **COMMON_LEARNER only** and combines both heldouts. Let residual TV mass and `TV > 0.35` tail share be aggregated into three broad regions:

- `ROOT`: `PREFLOP_ROOT`;
- `PREFLOP_CONTINUATION`: `PREFLOP_CONTINUATION_1 + PREFLOP_CONTINUATION_2PLUS`;
- `POSTFLOP`: `FLOP + TURN + RIVER`.

A broad region is called dominant only when **both** its residual-TV-mass share and its tail-state share are at least `0.35`. If multiple regions qualify, choose the region with the larger minimum of those two shares.

If no broad region qualifies, compute the top three scenario indices by COMMON pilot TV mass. If those three scenarios jointly contribute at least `0.50` of total pilot TV mass **and** at least `0.50` of all `TV > 0.35` states, classify the residual as `SCENARIO_CONCENTRATED`.

Otherwise classify it as `BROAD_MIXED_RESIDUAL`.

Frozen next routes:

- `ROOT_DOMINANT` -> `PRECOMMIT_ROOT_PREFLOP_ANCHOR_OR_LAGGED_BEHAVIOR_SCREEN`;
- `PREFLOP_CONTINUATION_DOMINANT` -> `PRECOMMIT_EARLY_PREFLOP_LAGGED_TARGET_OR_ANCHOR_SCREEN`;
- `POSTFLOP_DOMINANT` -> `LOCALIZE_POSTFLOP_BY_STREET_AND_SUPPORT_BEFORE_TRAINING`;
- `SCENARIO_CONCENTRATED` -> `PRECOMMIT_SCENARIO_STRATIFIED_CHANCE_SUPPORT_SCREEN`;
- `BROAD_MIXED_RESIDUAL` -> `REASSESS_REPRESENTATION_AND_VARIANCE_WITHOUT_MORE_GLOBAL_DAMPING`.

These are route selections only. Phase2B7 never authorizes training.

## 6. Validity / prohibitions

Phase2B7 is invalid if any of the following occurs:

- Phase2A or Phase2B6 result hash mismatch;
- wrong H2/3H source contract;
- wrong training/evaluation seeds;
- wrong first-1024 heldout slice;
- pilot policy artifact is not the completed Phase2B6 artifact with floor-training `0.25` and inference-floor `0.0`;
- reproduced Phase2B6 heldout metrics disagree beyond `1e-12` absolute tolerance;
- any policy evaluation applies a new floor or other smoothing at inference.

Prohibited:

- no traversal;
- no solver state generation;
- no reservoir insertion/replay;
- no optimizer steps;
- no model fitting/refitting;
- no heldout regeneration;
- no threshold relaxation;
- no seed shopping;
- no higher-floor training;
- no production authorization;
- no architecture winner selection;
- `READY FOR TABLES = NO`.

## 7. Strategic-strength firewall

The Phase2B6 candidate is not stability-eligible, so strategic-strength testing remains premature. Phase2B7 cannot change that. A separate precommitted comparison to the certified stable V1 control is mandatory only after a successor independently satisfies the stability eligibility gates.
