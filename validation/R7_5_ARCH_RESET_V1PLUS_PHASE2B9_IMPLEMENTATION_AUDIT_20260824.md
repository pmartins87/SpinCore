# R7.5 Architecture Reset — Phase2B9 Implementation Audit

Status: **IMPLEMENTED / FROZEN BEFORE OUTPUTS / RYZEN TESTS NOT YET RUN**  
Date: 2026-08-24

## Audit scope

Reviewed the Phase2B9 implementation against the frozen robust-Advantage-regression precommit and the completed Phase2B6/Phase2B8 evidence.

Files under audit:

- `validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B9_ROBUST_ADVANTAGE_REGRESSION_PRECOMMIT_20260824.md`
- `tools/r7_5_arch_reset_v1plus_phase2b9_robust_advantage_regression.py`
- `tools/test_r7_5_arch_reset_v1plus_phase2b9_robust_advantage_regression.py`
- `tools/run_r7_5_arch_reset_v1plus_phase2b9_robust_advantage_regression_ryzen.ps1`

Frozen evidence:

- exact Phase2B6 result SHA-256 `33ec6ba89823dae632b7af935def17444379c96a28e59478c0b7c91f1ec3659a`;
- exact Phase2B8 result SHA-256 `1fd9144a488cea6de0a7500320d552abf994908b5200146d4baa4bd6f81c4d98`;
- Phase2B8 equivalence-before-divergence PASS;
- exact Phase2B6 execution SHA `4fa96434321b59f63ed5c61f37de2c115c67315` is **not** the Phase2B6 pilot source and is intentionally not used here; the Phase2B6 pilot execution identity is `4fa96434321c32efc734a55ae75982018ff2d091`, which is what the implementation enforces.

## Findings

### 1. Phase2B9 is fit-only

The implementation never constructs a solver session and never calls `collect_root`, `collect_advantage_partial_exact`, or any other traversal entry point. It loads the exact completed Phase2B6 resume checkpoints and materializes their frozen Advantage reservoirs read-only.

No Strategy reservoir is mutated and no AveragePolicy model is fit.

### 2. Source checkpoint identity is hardened

For each training seed the source resume checkpoint must pass the canonical checkpoint loader with:

- H2 representation;
- THREE_HANDED domain;
- frozen action candidate;
- Phase2B6 pilot execution SHA `4fa96434321c32efc734a55ae75982018ff2d091`;
- frozen H2 architecture fingerprint;
- phase `phase2b6_resume`;
- iteration 3;
- global root 768;
- stage index 12;
- Phase2B6 checkpoint-extra schema;
- intervention floor exactly 0.25;
- exactly four stored final behavior-model states.

The launcher additionally verifies that each frozen Advantage reservoir is saturated at 100,000 retained items before any fit begins.

### 3. MSE and Huber are genuinely paired

For each seed/member pair, both models start from the same initialization seed and consume the same `random.Random(...).sample(...)` batch sequence for all 4096 optimizer steps.

The candidate changes only the Advantage regression loss. Optimizer, LR, batch size, legal mask, per-sample weights, model architecture, gradient clipping and source memory are shared.

Member 0 uses the canonical iteration-3 primary initialization seed plus a frozen Phase2B9 batch-seed namespace. Members 1-3 use the canonical iteration-3 side-member initialization and batch seeds.

### 4. Canonical side-member reproduction is an additional validity guard

Because Phase2B6 side members 1-3 were originally fit by the same deterministic independent-member routine, the paired MSE refit should reproduce those stored side-member parameters from the exact same final memory.

Phase2B9 computes the maximum absolute parameter difference for all six seed/member side-model checks and aborts if any exceeds `1e-7`. This is not a result-dependent admission threshold; it is an implementation-validity check proving that the paired fit path matches the existing canonical MSE path before the Huber comparison is trusted.

Member 0 is not required to reproduce the stored Phase2B6 primary model because the original primary fit intentionally consumed the live coupled batch RNG, whose pre-fit state is not reconstructed from the final checkpoint. The Phase2B9 member-0 MSE and Huber models nevertheless share an identical frozen batch sequence with each other.

### 5. The candidate loss is frozen

The candidate is the single precommitted Smooth-L1/Huber-style robust regression with transition scale `0.02`.

The implementation uses the standard Smooth-L1 algebra:

- quadratic region: `0.5 * error^2 / beta` for `|error| < beta`;
- linear region: `|error| - 0.5 * beta` otherwise.

`beta=0.02` is frozen from the independent Phase2B1 chance-noise scale before Phase2B9 outputs. No beta sweep exists in the code or launcher.

### 6. Behavior evaluation uses the canonical algebra

Heldout behavior is reconstructed from four raw Advantage-model outputs using `uncertainty_damped_policy_from_advantages` with the frozen:

- independent member regret matching;
- arithmetic mean member policy;
- ensemble-disagreement calculation;
- epsilon scale 1.75;
- epsilon cap 0.5;
- legal-action normalization.

Thus the readout compares the regression loss, not a new behavior-policy formula.

### 7. Region parsing is reused rather than reinvented

Phase2B9 reuses the authoritative Phase2B7 SPNNIV3 decoder for `PREFLOP_ROOT`, `PREFLOP_CONTINUATION_1`, `PREFLOP_CONTINUATION_2PLUS`, FLOP, TURN and RIVER classification.

The root and combined-continuation non-degradation checks therefore use the same region definitions that diagnosed the Phase2B6 residual.

### 8. Decision rule matches the precommit

A Huber causal-pilot eligibility PASS requires all of:

- both seed Huber ensemble NRMSE values <= 0.75;
- both heldout cross-seed mean TVs improve versus paired MSE;
- pooled improvement >= 0.03 absolute or >= 10% relative;
- paired bootstrap 95% CI strictly above zero;
- no heldout p95 degradation > 0.02;
- pooled dominant-action mismatch does not increase;
- pooled root mean TV does not worsen by > 0.01;
- pooled combined preflop-continuation mean TV does not worsen by > 0.01.

A PASS authorizes only a later Phase2B10 precommit. It does not authorize production training or select an architecture.

### 9. Resume semantics preserve expensive fits

Each completed paired member fit is stored under `ryzen_v1plus_phase2b9/seed_<seed>/paired_member_<member>.pt` and bound to:

- current diagnostic execution SHA;
- exact source checkpoint SHA-256;
- training seed/member;
- init/batch seeds;
- beta;
- step/batch/LR contract.

If interrupted, the launcher can be rerun without deleting the output directory; valid completed member fits are reused while invalid/mismatched artifacts are ignored and recomputed.

### 10. Governance is fail-closed

If the robust-regression screen fails, the code routes to `DESIGN_STRATIFIED_CHANCE_SUPPORT_OR_SOLVER_LEVEL_VARIANCE_REDUCTION`; it does not tune beta, revisit the failed lagged anchor, or increase the uniform floor.

If it passes, the next step is only `PRECOMMIT_PHASE2B10_HUBER_CAUSAL_TRAINING_PILOT`.

In either case:

- no production training is authorized;
- no architecture winner is selected;
- R7.5.4/R8 remain blocked;
- `READY FOR TABLES = NO`.

## Audit conclusion

The Phase2B9 implementation is consistent with the frozen mechanism test and is suitable for Ryzen execution. Its strongest validity properties are the exact Phase2B6 source binding, paired identical batches, canonical side-member MSE reproduction, and reuse of the canonical uncertainty behavior algebra.

The Ryzen launcher must still run `py_compile`, deterministic synthetic tests, source-memory preflight and the real paired fits before any Phase2B9 result is accepted.
