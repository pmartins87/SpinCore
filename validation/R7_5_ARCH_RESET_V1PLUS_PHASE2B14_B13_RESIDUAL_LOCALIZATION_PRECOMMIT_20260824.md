# R7.5 Architecture Reset — Phase2B14 B13 Residual Localization

Status: **FROZEN BEFORE PHASE2B14 OUTPUTS**  
Date: 2026-08-24

## 1. Purpose

Phase2B13 produced a statistically positive but sub-material root-IID64 effect under its frozen causal gate: COMMON pooled mean TV improved by `0.014520507258189153` (`6.922%`) with a strictly positive paired bootstrap CI, both heldouts improved, p95 improved, and NATIVE was non-contradictory. Because the precommitted materiality threshold was `>=0.02` absolute or `>=10%` relative, Phase2B13 remains a FAIL for scale-up.

Phase2B14 is a **read-only** localization. It asks where the remaining B13 candidate cross-seed instability lives, and whether the positive B13 effect is actually localized at the root that received the IID64 target intervention.

No training, solver traversal, target resampling, reservoir mutation, optimizer step, policy fitting, architecture selection, or production authorization is allowed.

## 2. Frozen inputs

- Phase2B13 uploaded result SHA-256: `6de7996282236d34adf5e8e53416fd8a443a1fbf5abc89fc807492d0cb3dbf80`;
- Phase2B13 execution SHA: `2cd7d1ece46a20d2b8937fe5135a415f6bbe54c2`;
- representation: `H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL`;
- domain: `THREE_HANDED`;
- arms: `IID1_OF_64_EQUAL_COMPUTE_CONTROL`, `IID64_MEAN_CANDIDATE`;
- learner modes: `COMMON_LEARNER`, `NATIVE_LEARNER`;
- training seeds: `1342191342`, `1801739323`;
- evaluation seeds: `2029384436`, `1150634112`;
- exactly the first `1024` frozen heldout descriptors per evaluation seed, matching Phase2B13.

The exact local Phase2B13 policy artifacts and their metadata must be hash-validated before inference.

## 3. Mandatory reproduction gate

Before any localization is interpreted, Phase2B14 must reproduce from the saved Phase2B13 policies, within absolute tolerance `1e-12`, all eight heldout mean/p95 values recorded in Phase2B13: control and candidate, COMMON and NATIVE, both evaluation seeds.

Any mismatch -> `PHASE2B14_INVALID_REPRODUCTION_STOP`.

## 4. Frozen region taxonomy

Decode authoritative SPNNIV3 history exactly as in Phase2B7:

- `PREFLOP_ROOT`: street 0 and zero non-forced preflop history events;
- `PREFLOP_CONTINUATION_1`: street 0 and one non-forced preflop event;
- `PREFLOP_CONTINUATION_2PLUS`: street 0 and at least two non-forced preflop events;
- `FLOP`, `TURN`, `RIVER` from the authoritative street byte.

Also report axes for actor, scenario index, action-path length, legal-action count and history-count bin.

## 5. Primary residual readout

Primary learner: `COMMON_LEARNER`.

For every heldout state compute:

- control cross-seed TV;
- candidate cross-seed TV;
- signed improvement `control_tv - candidate_tv`;
- candidate tail indicator `TV > 0.35`.

For each region report control/candidate mean, p50, p95, candidate TV mass share, candidate tail share, and mean improvement.

Broad regions:

- `ROOT` = `PREFLOP_ROOT`;
- `PREFLOP_CONTINUATION` = continuation 1 + continuation 2+;
- `POSTFLOP` = FLOP + TURN + RIVER.

A broad region is residual-dominant only if both candidate TV-mass share and candidate tail share are `>=0.35`, reusing the Phase2B7 dominance threshold.

## 6. Root-effect consistency check

The root IID64 intervention is considered directionally localized only if, for COMMON learner:

1. candidate root mean TV is lower than control root mean TV on evaluation seed `2029384436`;
2. candidate root mean TV is lower than control root mean TV on evaluation seed `1150634112`;
3. pooled root mean improvement is strictly positive.

This check is descriptive only and cannot retroactively change the Phase2B13 FAIL.

## 7. Frozen route classification

After reproduction passes:

- continuation residual dominant **and** root effect consistent -> `PREFLOP_CONTINUATION_RESIDUAL_DOMINANT_AFTER_ROOT_IID64`; next route `PRECOMMIT_POSTERIOR_WEIGHTED_PREFLOP_CONTINUATION_CHANCE_SCREEN`;
- continuation residual dominant but root effect inconsistent -> `PREFLOP_CONTINUATION_DOMINANT_ROOT_EFFECT_NOT_LOCALIZED`; next route `REASSESS_REPRESENTATION_SUPPORT_BEFORE_MORE_CHANCE_INTEGRATION`;
- root residual dominant -> `ROOT_RESIDUAL_DOMINANT_AFTER_IID64`; next route `REASSESS_ROOT_ESTIMATOR_OR_REPRESENTATION_SUPPORT_NO_SCALEUP`;
- postflop residual dominant -> `POSTFLOP_RESIDUAL_DOMINANT_AFTER_ROOT_IID64`; next route `LOCALIZE_POSTFLOP_SUPPORT_BEFORE_ANY_NEW_TRAINING`;
- if no broad region dominates but the top three scenarios jointly carry at least 50% of both candidate TV mass and tail -> `SCENARIO_CONCENTRATED_RESIDUAL_AFTER_ROOT_IID64`; next route `PRECOMMIT_SCENARIO_STRATIFIED_SUPPORT_SCREEN`;
- otherwise -> `BROAD_MIXED_RESIDUAL_AFTER_ROOT_IID64`; next route `REASSESS_REPRESENTATION_SUPPORT_AND_VARIANCE_NO_SCALEUP`.

## 8. Guardrails

- Phase2B13 materiality gate remains failed; no reinterpretation of `0.01452` as a causal PASS;
- no x4 confirmation;
- no K128/K256;
- no higher preflop floor;
- no naive uniform hidden-card resampling after observed actions;
- no training based only on Phase2B14;
- no gate relaxation or seed shopping;
- no architecture winner selection;
- `PRODUCTION TRAINING = NO`;
- `READY FOR TABLES = NO`.

If the continuation route is selected, any next chance estimator must account for the posterior over hidden cards conditional on the observed betting history; Phase2B14 itself does not specify or authorize that estimator.
