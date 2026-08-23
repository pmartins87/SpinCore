# R7.5 Architecture Reset — Phase2B6 Implementation Audit

Status: **IMPLEMENTED / FROZEN BEFORE OUTPUTS / RYZEN TESTS NOT YET RUN**  
Date: 2026-08-22

## Audit scope

Reviewed the Phase2B6 implementation against the precommitted causal contract and the authoritative Phase2A/x4 training semantics.

Files under audit:

- `validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B6_PREFLOP_DAMPING_TRAINING_PILOT_PRECOMMIT_20260822.md`
- `tools/r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot.py`
- `tools/test_r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot.py`
- `tools/run_r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot_ryzen.ps1`

Reference semantics audited:

- `tools/r7_5_3d_v1plus_phase2a_strategy_capacity.py`
- `tools/r7_5_3d_v1plus_phase2a_strategy_capacity_runtimefix.py`
- `tools/r7_5_3d_v1plus_phase2a_policy_fit_worker.py`
- `tools/r7_5_3c_chance_coverage_x4_domain_worker_runtimefix.py`
- `python/spincore/r7_5_action_cfr.py`
- `python/spincore/r7_5_representation_v3.py`
- `python/spincore/r7_5_representation_v3_stage.py`
- `python/spincore/r7_5_representation_v3_stage_contract.py`

## Findings

### 1. Intervention scope matches Phase2B5

The wrapper first obtains and validates the existing uncertainty-damped native behavior policy. It applies

`0.75 * native + 0.25 * uniform(legal)`

only when the authoritative SPNNIV3 observation reports `street == PREFLOP` and at least one non-forced preflop event. Root preflop states with zero voluntary events remain native, and all postflop states remain native.

The parser validates the SPNNIV3 magic, exact `120 + 20 * history_count` wire length, street range, and event-level `street/forced` fields before applying the intervention.

### 2. The floor affects the same recursion surface screened in Phase2B5

`UniversalPartialExactCollector._adv_partial` queries its policy at every nonterminal decision, including traverser nodes used for the regret baseline and opponent nodes used for exact/sampled continuation. `collect_strategy_own_reach` also queries the same policy for Strategy-memory targets and target-player own-reach sampling.

Phase2B6 replaces only `session.collector.policy` with a wrapper around the live ensemble object. The ensemble itself remains the object updated after every Advantage fit. Therefore the intervention changes the intended training behavior without replacing the Advantage model lifecycle.

### 3. Iteration-1 neutrality is preserved

Before the first Advantage fit the native ensemble has no models and returns the exact uniform legal policy. Mixing a uniform policy with the same uniform distribution leaves it unchanged. Thus the intervention cannot create an artificial difference before the first learned Advantage ensemble exists.

The synthetic test explicitly checks this algebraic neutrality.

### 4. Phase2A x4 chance/fit schedule is preserved

Phase2B6 uses:

- 3 iterations;
- 4 chunks/iteration;
- 64 roots/chunk;
- 768 roots/seed;
- the same `scenario_index = global_root % 15` schedule;
- the same frozen `deck_seed(training_seed, global_root, iteration)`;
- exact-opponent depth 2;
- the same Advantage reservoir, model-reset seeds, four-member ensemble, optimizer, 4096 steps/member, batch size, learning rate, and audits.

It directly uses the previously audited x4 `_fit_only_iteration` helper after each set of four chunks, avoiding the historical zero-root reporting division bug without changing fit semantics.

### 5. Resume semantics are causal-safe

A Phase2B6 checkpoint is written after every chunk and is tied to the current execution SHA, H2/3H identity, action candidate, architecture fingerprint, exact stage config, floor value, stage index, bundle reservoirs/RNG state, behavior ensemble states, behavior statistics, floor telemetry, and pending iteration state.

Resume accepts only the Phase2B6 checkpoint schema/phase and exact execution SHA. Phase2A artifacts are never used as mutable resume state.

### 6. Final learner comparison matches the Phase2A causal pattern

The Phase2B6 100k Strategy memory is fitted with both:

- `COMMON_LEARNER`: exact Phase2A common initialization and common batch RNG;
- `NATIVE_LEARNER`: exact seed-coupled initialization and the authoritative post-trajectory bundle batch-RNG state.

Both use the authoritative policy audit seed `training_seed ^ 0x71A5BEEF`, 16384 steps, batch 256, LR 0.001, and the existing Phase2A `_fit_policy` implementation.

### 7. Heldout evaluation does not apply the floor

The pilot policy artifacts contain only learned AveragePolicy model state. Heldout probabilities are generated directly from those models with no behavior wrapper. The canonical variable-length `legal_slots` -> ten-slot `legal_mask` correction from the Phase2A evaluation recovery is implemented explicitly before SPNNIV3 collation.

Thus a favorable heldout result cannot be created merely by flattening both evaluated seed policies after training.

### 8. Exact Phase2A control is reused and identity-checked

The tool requires the exact recovered Phase2A result SHA-256 `65f691e6...` and extracts the four `S100K_CONTROL` policy artifact hashes from its completed-source inventory. Every local baseline policy file is rehashed before evaluation.

No new native control is trained.

### 9. Decision gates are precommitted and encoded

The implementation requires:

- local Advantage and both final-policy learner fit gates;
- COMMON practical improvement >=0.02 absolute or >=10% relative;
- strictly positive 95% paired stratified-bootstrap lower bound;
- improvement on both COMMON heldout seeds;
- COMMON p95 degradation <=0.02 per heldout seed;
- NATIVE pooled non-worsening and <=0.01 mean degradation per heldout seed.

Only after those causal conditions does it inspect the historical hard stability gates (`mean<=0.15`, `p95<=0.35`) on both COMMON heldouts.

No failure path escalates automatically to a 50%, 75%, or 100% floor.

## Ryzen execution gate

The launcher itself rejects any tracked worktree modification while intentionally ignoring the user's local untracked artifact directories/files. It then verifies the frozen Python/Torch/Numpy environment, compiles the pilot/test/reference Python scripts, runs deterministic synthetic tests, validates exact Phase2A and Phase2B5 result hashes, validates all four exact Phase2A S100K baseline policy hashes, validates the frozen H2/3H contract, rebuilds/validates the x64 solver, and only then starts the two-seed training pilot.

The synthetic tests and Windows execution have **not** been claimed PASS here; they will be executed by the Ryzen launcher before any training starts.

## Audit conclusion

The implementation is consistent with the Phase2B6 precommit and isolates the intended causal question with no identified contract broadening. It is appropriate to run the one small 25% preflop-continuation training pilot.

This audit does not change governance: old R7.5.3 remains closed, architecture winner is unset, production training remains unauthorized, strategic strength remains unevaluated, and `READY FOR TABLES = NO`.
