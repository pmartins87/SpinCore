# R7.5 Architecture Reset — Phase2C1 Implementation Audit

Date: 2026-08-24
Status: **IMPLEMENTED / FROZEN BEFORE OUTPUTS / RYZEN TESTS NOT YET RUN**

## Source gate

Phase2C1 requires the exact successful Phase2C0 result SHA256
`55e83be4fd8776e0fcdb63e7d4400ed05aff8c48213898ad8f1abe3713a35876`, status
`STRUCTURAL_REACH_FACTORIZATION_FEASIBLE`, `screen_pass=true`, and frozen next route
`PRECOMMIT_PHASE2C1_EXACT_RANGE_REACH_SOLVER_PROTOTYPE`.

Phase2C0 evidence is persisted in
`validation/R7_5_ARCH_RESET_V1PLUS_PHASE2C0_RESULT_EVIDENCE_20260824.json`.

## What was implemented

`tools/r7_5_arch_reset_v1plus_phase2c1_exact_range_reach_solver_prototype.py` constructs an explicit structural
reach state for each of the same 8 Phase2C0 heldout anchors under each of the two frozen Phase2B13 behavior seeds.
For the final actor's fixed private cards it keeps two 2,450-entry float64 opponent reach vectors and propagates
them event-by-event through the observed public preflop action path.

When an opponent acts, only that opponent's reach vector is multiplied by the frozen behavior probability of the
observed action for each candidate private hand. When the final actor previously acted, only the known-hand scalar
is updated. Public action application remains delegated to the authoritative solver.

This is not posterior importance weighting, rejection sampling, target averaging, or a new chance estimator.
No future board or opponent hand is sampled by the reach update.

## Exactness guards

For every task:

* final SPNNIV3, actor, active mask, and legal slots must match the heldout target exactly;
* 128 deterministic hands per opponent are compared against the original Phase2C0 full-history seat likelihood;
* the known-actor scalar is compared against the original direct actor component;
* collision-masked joint normalizer, effective support, and positive assignment count are recomputed from the
  incremental reach vectors and compared to the exact matching Phase2C0 row;
* raw reach storage is counted as two `2450 × float64` vectors = 39,200 bytes;
* structural policy-evaluation count is recorded and must not exceed 4,902 per task.

The final gates are frozen at absolute/relative tolerance `1e-12` as specified in the precommit.

## Runtime preflights

The Ryzen launcher:

1. requires a clean tracked worktree while intentionally ignoring untracked local artifacts;
2. verifies Python 3.11.9, 64-bit runtime, torch 2.13.0+cpu and numpy 2.3.5;
3. py-compiles the C1/C0/runtime-fix scripts and runs deterministic C1 synthetic tests;
4. verifies the exact local Phase2C0 result SHA/status/route;
5. fresh-builds the authoritative x64 solver DLL with VS2022;
6. verifies PE x64, ABI2, SPNNIV3 and explicit-deal diagnostic symbols;
7. reruns the Phase2B10 explicit-deal round-trip suite;
8. reruns the Phase2B15 all-64-anchor Windows canonical heldout reconstruction preflight;
9. only then starts the 16 structural prototype tasks under the frozen-run evidence wrapper.

No real Windows/Ryzen PASS is claimed by this audit. Those checks have not yet been executed on the user's Ryzen.

## Finite decision firewall

PASS permits only one separately precommitted bounded range/reach target-kernel causal pilot. FAIL selects the
certified stable V1 fallback and closes the V1+ architecture reset. Neither outcome from Phase2C1 itself authorizes
training, full-x4 confirmation, architecture winner selection, production training, or table use.

## Files

* `validation/R7_5_ARCH_RESET_V1PLUS_PHASE2C1_EXACT_RANGE_REACH_SOLVER_PROTOTYPE_PRECOMMIT_20260824.md`
* `tools/r7_5_arch_reset_v1plus_phase2c1_exact_range_reach_solver_prototype.py`
* `tools/test_r7_5_arch_reset_v1plus_phase2c1_exact_range_reach_solver_prototype.py`
* `tools/run_r7_5_arch_reset_v1plus_phase2c1_exact_range_reach_solver_prototype_ryzen.ps1`
* `validation/R7_5_ARCH_RESET_V1PLUS_PHASE2C0_RESULT_EVIDENCE_20260824.json`
