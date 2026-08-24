# R7.5 Architecture Reset — Phase2B14 Implementation Audit

Status: **IMPLEMENTED / FROZEN BEFORE OUTPUTS / RYZEN TESTS NOT YET RUN**  
Date: 2026-08-24

## Scope

Audited:

- `validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B13_RESULT_EVIDENCE_20260824.json`;
- `validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B14_B13_RESIDUAL_LOCALIZATION_PRECOMMIT_20260824.md`;
- `tools/r7_5_arch_reset_v1plus_phase2b14_b13_residual_localization.py`;
- `tools/test_r7_5_arch_reset_v1plus_phase2b14_b13_residual_localization.py`;
- `tools/run_r7_5_arch_reset_v1plus_phase2b14_b13_residual_localization_ryzen.ps1`;
- inherited Phase2B7 SPNNIV3 region decoder/grouping;
- inherited Phase2B13 policy loading and Phase2B6 heldout inference helpers.

## Findings

1. **Phase2B13 FAIL is immutable.** The tool requires exact result SHA `6de7996282236d34adf5e8e53416fd8a443a1fbf5abc89fc807492d0cb3dbf80`, exact status `ROOT_IID64_TRAINING_EFFECT_NOT_SUPPORTED`, `causal_effect_supported=false`, `common_materiality_pass=false`, and the frozen no-scaleup route. Phase2B14 cannot reinterpret the positive CI as a causal PASS.

2. **Saved policy identities are checked before inference.** For both arms, both training seeds and both learner modes, the implementation validates seed-result identity, policy metadata identity, K=64, execution SHA, and policy artifact SHA-256 before loading the network.

3. **The mandatory reproduction gate is fail-closed.** The tool recomputes control/candidate cross-seed heldout mean and p95 for COMMON and NATIVE on both evaluation seeds and requires all eight values to reproduce the Phase2B13 result within `1e-12` absolute error. Any mismatch aborts before localization.

4. **No solver/chance/training path exists in Phase2B14.** The implementation only loads heldout descriptors and saved AveragePolicy networks. It does not construct a solver, traverse a tree, resample hidden cards, insert a reservoir sample, fit Advantage, fit AveragePolicy, or mutate a Phase2B13 artifact.

5. **Region semantics are inherited from the already-audited Phase2B7 decoder.** Initial root, first preflop continuation, deeper preflop continuation and postflop street classification use authoritative SPNNIV3 bytes and exact non-forced-preflop history counting.

6. **Residual mass and tail use candidate cross-seed TV.** The candidate is the B13 IID64 arm. Broad dominance reuses the Phase2B7 threshold: both TV-mass share and `TV>0.35` tail share must be at least `0.35`.

7. **Root-effect consistency is directional and descriptive.** It requires lower candidate root mean TV on each heldout and positive pooled root improvement. It does not authorize x4 or alter the B13 materiality decision.

8. **Continuation chance integration is not implemented here.** If the frozen route selects a continuation-chance screen, that later experiment must account for the hidden-card posterior conditional on observed betting history. Naive uniform resampling after actions remains prohibited.

9. **Synthetic tests cover route logic.** They check continuation-dominant/root-helped, continuation-dominant/root-not-consistent, root-dominant, and frozen constants. Real artifact reproduction remains a Ryzen preflight/runtime gate.

10. **Strategic firewall remains intact.** Phase2B14 has `training_authorized=false`, `full_x4_confirmation_authorized=false`, `architecture_winner_selected=false`, `production_training_authorized=false`, and `ready_for_tables=false`.

## Audit conclusion

The implementation matches the frozen Phase2B14 read-only question and is suitable for Ryzen execution subject to real `py_compile`, synthetic tests, exact B13 result verification, H2/3H contract validation, local policy hash validation, and the `1e-12` heldout reproduction gate.

No Phase2B14 output existed at audit time. `PRODUCTION TRAINING = NO`; `READY FOR TABLES = NO`.
