# R7.5 Architecture Reset — Phase2B10 Implementation Audit

Status: **IMPLEMENTED / FROZEN BEFORE OUTPUTS / RYZEN TESTS NOT YET RUN**  
Date: 2026-08-24

## Audit scope

Reviewed the Phase2B10 private/public chance decomposition against the frozen precommit and the existing solver/training contracts.

Files under audit:

- `validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B10_PRIVATE_PUBLIC_CHANCE_DECOMPOSITION_PRECOMMIT_20260824.md`
- `tools/r7_5_arch_reset_v1plus_phase2b10_private_public_chance_decomposition.py`
- `tools/test_r7_5_arch_reset_v1plus_phase2b10_private_public_chance_decomposition.py`
- `tools/run_r7_5_arch_reset_v1plus_phase2b10_private_public_chance_decomposition_ryzen.ps1`
- `include/spincore/hand_engine.hpp`
- `src/hand_engine.cpp`
- `include/spincore/spin_traversal_state.hpp`
- `src/spin_traversal_state.cpp`
- `include/spincore/solver_c_api.h`
- `src/solver_c_api.cpp`
- `python/spincore/solver.py`

Frozen evidence:

- Phase2B1 result SHA-256 `f95751afeb17fcd5844bfcb2971577b92a400750444e5dabe2f4ddb5718ba6ef`;
- Phase2B6 result SHA-256 `33ec6ba89823dae632b7af935def17444379c96a28e59478c0b7c91f1ec3659a`;
- Phase2B9 result SHA-256 `71f77b6921597c7b1d048f8fb3e448f5fce74a974b247ac4ca88383fece5c64a`;
- Phase2B1 source classification `CHANCE_DOMINANT`;
- Phase2B9 route `DESIGN_STRATIFIED_CHANCE_SUPPORT_OR_SOLVER_LEVEL_VARIANCE_REDUCTION`.

## Findings

### 1. The diagnostic separates chance at a valid boundary

Phase2B10 performs the private/public decomposition only at the initial preflop root, before any voluntary action history exists. This is important: resampling opponent private cards at a later observed public history would require action-likelihood/posterior correction. At the initial root, conditioning only on the acting player's private cards is exact and does not introduce that posterior-history bias.

### 2. Explicit-deal support is additive and fail-closed

The existing seed-based solver constructor is unchanged. The ABI version remains 2 and two new additive diagnostic entry points are exposed:

- explicit root creation from validated card ids;
- read-only deal snapshot.

The Python binding detects the extension conditionally, so older ABI-v2 solver binaries remain usable by historical code. Phase2B10 itself requires the extension and aborts if it is absent.

### 3. Explicit deals are validated twice

The Python wrapper validates:

- three two-card seat rows;
- five board ids;
- dead-seat `-1` convention;
- live/board card range `0..51`;
- global card uniqueness.

The C++ `HandEngine` explicit constructor independently validates live/dead seat assignment, card validity and uniqueness. Thus malformed diagnostic deals cannot silently enter traversal.

### 4. Seed-created deal round-trip is tested

The Phase2B10 test suite exports a seeded root deal, recreates the root from explicit cards, and requires exact equality of:

- deal snapshot;
- SPNNIV3 root bytes;
- actor;
- universal legal identity.

It then applies the same effective universal action to both states and requires identical next-state SPNNIV3 bytes or identical terminal chip delta. This tests that explicit creation changes only how chance is supplied, not betting-state semantics.

### 5. Chance arms preserve the intended conditionals

For every anchor:

- `TRAVERSAL_ONLY` preserves the full deal;
- `PRIVATE_ONLY` preserves actor holes and all five board cards while uniformly resampling the four opponent-hole slots;
- `PUBLIC_ONLY` preserves all six player hole cards while uniformly resampling the ordered five-card board;
- `COMBINED` preserves actor holes while uniformly resampling opponent holes and board.

All card draws are without replacement from the exact remaining deck. No semantic hand bucket or result-dependent filtering is used.

### 6. Every variant must preserve the exact acting-player root infoset

Before a target is accepted, the worker requires the explicit variant to reproduce the stored Phase2B1:

- SPNNIV3 observation SHA-256;
- actor;
- universal legal tuple;
- legal mask.

Any hidden/public resampling that leaks into the acting player's root observation aborts the diagnostic.

### 7. The current best causal behavior is used

The diagnostic uses the exact final Phase2B6 four-member Advantage behavior states for each source training seed and wraps them with the exact Phase2B6 behavior intervention:

- root preflop native;
- continuation preflop 25% uniform floor;
- postflop native.

This makes the decomposition relevant to the best causally supported successor rather than reverting to the inferior Phase2A control or failed Phase2B8/Phase2B9 candidates.

### 8. Source checkpoints are loaded read-only

Phase2B10 validates for each Phase2B6 checkpoint:

- H2 / THREE_HANDED;
- exact action candidate;
- Phase2B6 execution SHA `4fa96434321c32efc734a55ae75982018ff2d091`;
- frozen architecture fingerprint;
- phase `phase2b6_resume`;
- iteration 3;
- global root 768;
- stage 12;
- exact Phase2B6 checkpoint-extra schema;
- floor 0.25;
- four final behavior states.

No memory or checkpoint field is mutated.

### 9. Traversal/action RNG remains separately controlled

`TRAVERSAL_ONLY` varies traversal RNG while the full deal is fixed. All three chance arms hold traversal RNG fixed within each scenario/anchor task. Therefore private/public changes are not intentionally confounded with action-sampling changes.

### 10. The work is parallelized at the natural independent-task level

The frozen workload is:

`2 behaviors × 15 scenarios × 4 anchors × 4 arms × 8 replicates = 3,840 root target traversals`.

The launcher uses 30 independent worker processes with one Torch/OMP/MKL/OpenBLAS thread each, leaving a small amount of host capacity for Windows/process orchestration. Worker count is compute-only and does not alter seeds, deals, pairings, or gates.

Unlike Phase2B9, Phase2B10 has 240 independent tasks per source behavior seed, so broad process-level parallelism is scientifically safe and computationally useful.

### 11. Decision rule matches the precommit

The implementation computes K1 non-overlapping target pairs and reports target MAD, sign disagreement, regret-matching TV and dominant-action mismatch.

The classification uses the frozen `0.10` material excess-TV threshold and `1.5x` dominance ratio, plus same-direction confirmation in both source behavior seeds for a single-component dominant classification.

No classification directly authorizes training.

### 12. Governance remains fail-closed

Depending on the readout, the only routes are:

- public-board chance design;
- private-hole stratified chance design;
- factorized mixed chance design;
- or representation/support reassessment if unresolved.

There is no route back to Huber beta tuning, lag-anchor tuning, or higher continuation floor.

In all cases:

- no architecture winner is selected;
- no production training is authorized;
- R7.5.4/R8 remain blocked;
- `READY FOR TABLES = NO`.

## Audit conclusion

The Phase2B10 implementation is consistent with the frozen causal question and is suitable for Ryzen execution **subject to the launcher's real CMake/MSVC build and explicit-deal round-trip tests passing**. The main scientific advantage of this phase is that it separates opponent-private and future-public chance at a root boundary where the conditional resampling is exact, avoiding the posterior-history problem that would arise from naively reseeding hidden cards inside an already-observed betting history.
