# R7.5 Architecture Reset — Phase2B7 Implementation Audit

Status: **IMPLEMENTED / FROZEN BEFORE OUTPUTS / RYZEN TESTS NOT YET RUN**  
Date: 2026-08-23

## Audit scope

Reviewed Phase2B7 against the frozen residual-localization precommit and the completed Phase2A/Phase2B6 evaluation semantics.

Files under audit:

- `validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B7_RESIDUAL_LOCALIZATION_PRECOMMIT_20260823.md`
- `tools/r7_5_arch_reset_v1plus_phase2b7_residual_localization.py`
- `tools/test_r7_5_arch_reset_v1plus_phase2b7_residual_localization.py`
- `tools/run_r7_5_arch_reset_v1plus_phase2b7_residual_localization_ryzen.ps1`

Reference sources:

- exact Phase2A result/evidence and `S100K_CONTROL` policy artifacts;
- exact Phase2B6 result/evidence and completed pilot policy artifacts;
- `tools/r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot.py` for canonical model loading, heldout discovery, legal-mask conversion and policy probabilities;
- `python/spincore/r7_5_representation_v3_referee_artifacts.py` for frozen heldout descriptors;
- `python/spincore/r7_5_representation_v3_stage_contract.py` for H2/3H identity and historical gates.

## Findings

### 1. Diagnostic is read-only

The Phase2B7 tool loads existing Phase2A and Phase2B6 policy state dicts and evaluates them on existing heldout descriptors. It has no solver-library argument, no traversal/session construction, no reservoir API, no optimizer, and no fitting loop. The launcher does not build or invoke the solver.

### 2. Exact input identity is enforced

The tool requires the exact Phase2A result SHA-256 `65f691e6...` and Phase2B6 result SHA-256 `33ec6ba8...`. It validates the Phase2B6 execution SHA `4fa9643...`, the supported-causal-effect status, the frozen next route, the H2/3H source/model contract, the exact Phase2A S100K baseline artifacts, the local Phase2B6 seed results, and each Phase2B6 policy metadata/artifact hash pair.

The heldout gzip files are also rehashed against the hashes embedded in the exact Phase2B6 result.

### 3. Phase2B6 metrics must be reproduced before localization

For each learner mode and heldout seed, Phase2B7 recomputes the Phase2B6 cross-seed AveragePolicy mean and p95 TV from the exact model artifacts. The diagnostic aborts if either differs from the recorded Phase2B6 result by more than `1e-12` absolute tolerance.

This prevents residual analysis from silently using different models, heldouts, legal masks or inference behavior.

### 4. SPNNIV3 causal-region parser matches the training boundary

The parser validates the `SPNNIV3\0` magic, exact wire length `120 + 20 * history_count`, street byte, and every history event used for the non-forced-preflop count. The primary regions exactly match the precommit:

- preflop root;
- preflop continuation after one voluntary event;
- preflop continuation after two or more voluntary events;
- flop;
- turn;
- river.

The same parser boundary was used by Phase2B6 to decide where the 25% floor was active.

### 5. Secondary partitions are frozen and implemented

The tool reports actor, scenario index, action-path-length bin, legal-action count, and total-history-count bin exactly as frozen before output inspection. No additional post-hoc partition is encoded.

### 6. Residual and tail accounting are explicit

Every group reports baseline/pilot distribution summaries, mean improvement, residual TV mass share, and contribution to the `TV > 0.35` tail. The `0.35` value is the existing historical p95 stability threshold, reused only to localize the tail rather than introduced as a new decision gate.

### 7. Routing logic matches the precommit

The implementation aggregates COMMON residuals into root, preflop-continuation and postflop broad regions. Dominance requires both residual-TV-mass share and tail share `>= 0.35`. If no broad region qualifies, the frozen top-three-scenario `>= 0.50` mass-and-tail rule is evaluated; otherwise the result is broad/mixed.

The route output never authorizes training. It explicitly retains `higher_floor_training_authorized=false`, `architecture_winner_selected=false`, `production_training_authorized=false`, and `ready_for_tables=false`.

### 8. Synthetic tests cover parser and every routing branch

The deterministic test file covers:

- all six causal-region labels;
- path/history bins;
- residual group accounting;
- `ROOT_DOMINANT`;
- `PREFLOP_CONTINUATION_DOMINANT`;
- `POSTFLOP_DOMINANT`;
- `SCENARIO_CONCENTRATED` under deliberately non-dominant broad-region shares;
- `BROAD_MIXED_RESIDUAL`.

The test design was corrected before any Phase2B7 real output so the scenario-concentration and broad-mixed fixtures do not accidentally satisfy the postflop-dominance rule.

The tests have **not** been claimed PASS on the Ryzen here; the launcher runs `py_compile` and the deterministic test suite before the real readout.

## Audit conclusion

The implementation is consistent with the frozen Phase2B7 contract and is suitable for a cheap read-only Ryzen readout. It should answer whether the remaining Phase2B6 instability is concentrated at root preflop, later preflop continuation, postflop, a small scenario subset, or broadly distributed.

No training, higher damping, architecture selection, production authorization or table deployment is justified by this audit. `READY FOR TABLES = NO`.
