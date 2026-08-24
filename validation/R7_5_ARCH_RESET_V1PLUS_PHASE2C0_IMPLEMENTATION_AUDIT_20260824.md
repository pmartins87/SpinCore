# R7.5 Architecture Reset — Phase2C0 Implementation Audit

Date: 2026-08-24
Status: IMPLEMENTED / FROZEN BEFORE OUTPUTS / RYZEN TESTS NOT YET RUN

## Why Phase2C0 exists

Phase2B16 closed the estimator-repair path. Exact rejection posterior sampling materially improved the failed Phase2B15 SNIS estimator but still failed the precommitted pooled mean-TV ceiling (0.2517340306655365 > 0.24). The next permitted choice was structural reach support or certified stable V1 fallback. Because every other B16 support gate passed and the miss was limited to the mean-TV ceiling, one bounded structural feasibility audit is justified before fallback.

## Scope

Phase2C0 performs no Advantage target traversal, network fit, optimizer step, reservoir mutation, AveragePolicy fit, x4 confirmation, production training, or table-use action. It evaluates whether preflop action-history likelihood can be represented exactly as two opponent private-hand reach tables plus card-removal coupling.

## Ordered-hand correctness

SPNNIV3 retains the two current-player private-card slots separately. Phase2C0 therefore does not assume that swapping the two hole-card slots is a proven symmetry. With the current actor's two cards fixed, each opponent table enumerates 50×49 = 2,450 ordered private hands. The exact joint prior support across two distinguishable opponent seats is 50×49×48×47 = 5,527,200 ordered assignments.

## Factorization logic

At every observed public preflop action, the frozen behavior probability is computed from the acting player's SPNNIV3 observation. That observation contains the acting player's private cards plus public state/history, never another player's hidden cards. Therefore the full path likelihood should factor into the product of seat-local action-likelihood components, with only card removal coupling the two opponent private hands.

The implementation does not assume this proof is sufficient. It performs real solver checks on 8 deterministic heldout anchors × 2 source behavior seeds:

* two complete 2,450-hand opponent likelihood tables per task;
* 128 deterministic valid joint assignments per task comparing full path likelihood with the factorized product;
* 32 deterministic candidate hands per opponent replayed with an alternate filler opponent hand/board to verify seat-component independence;
* exact SPNNIV3/actor/active-mask/legal identity on every replay;
* exact collision-masked joint normalizer and posterior effective support computed from the two tables.

## Windows and source identity

The audit inherits the Phase2B15 canonical suit-isomorphic Windows heldout reconstruction. Historical Linux `deck_seed` is not used to reconstruct the heldout deal on Windows. The launcher reruns the all-64-anchor reconstruction preflight before Phase2C0.

The exact Phase2B16 result SHA required is `3b5e71c3cc92ed530589877f6790333b1f94b579bb39e7c687082787693d958c` and its status must be `EXACT_POSTERIOR_STILL_TOO_UNSTABLE_CLOSE_ESTIMATOR_REPAIR_PATH`.

## Frozen decision

PASS requires all 16 tasks, factorization and filler-independence errors <=1e-12, positive finite normalizers, and <=4,901 seat-policy table evaluations per task. PASS permits only a separately precommitted Phase2C1 exact range/reach solver prototype.

FAIL selects the certified stable V1 fallback and closes the V1+ architecture reset. No post-result factorization threshold tuning or return to estimator repair is allowed.

## Files

* `validation/R7_5_ARCH_RESET_V1PLUS_PHASE2C0_STRUCTURAL_REACH_FACTORIZATION_PRECOMMIT_20260824.md`
* `tools/r7_5_arch_reset_v1plus_phase2c0_structural_reach_factorization.py`
* `tools/test_r7_5_arch_reset_v1plus_phase2c0_structural_reach_factorization.py`
* `tools/run_r7_5_arch_reset_v1plus_phase2c0_structural_reach_factorization_ryzen.ps1`
* `validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B16_RESULT_EVIDENCE_20260824.json`

Real Windows/Ryzen build, synthetic tests, heldout reconstruction, and factorization outputs have not yet been run and are not claimed as PASS in this audit.
