# R7.5 Architecture Reset — Phase2B8 Implementation Audit

Status: **IMPLEMENTED / FROZEN BEFORE OUTPUTS / RYZEN TESTS NOT YET RUN**  
Date: 2026-08-23

## Audit scope

Reviewed the Phase2B8 implementation against the frozen `LAGGED_BEHAVIOR_ANCHOR_025` precommit and the completed Phase2B6/Phase2B7 evidence.

Files under audit:

- `validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B8_LAGGED_PREFLOP_ANCHOR_PRECOMMIT_20260823.md`
- `tools/r7_5_arch_reset_v1plus_phase2b8_lagged_preflop_anchor.py`
- `tools/test_r7_5_arch_reset_v1plus_phase2b8_lagged_preflop_anchor.py`
- `tools/run_r7_5_arch_reset_v1plus_phase2b8_lagged_preflop_anchor_ryzen.ps1`

Reference inputs:

- exact Phase2B6 result SHA-256 `33ec6ba89823dae632b7af935def17444379c96a28e59478c0b7c91f1ec3659a`;
- exact Phase2B7 result SHA-256 `ff55a5a047d62952e505b8e4d59d79d4016f30b6696a339318bc696dd6f77fe6`;
- exact four completed Phase2B6 control AveragePolicy artifacts frozen inside Phase2B7;
- exact two frozen heldout artifacts frozen inside Phase2B7;
- the Phase2B6 collector/deck schedule and fit-only runtime path.

## Findings

### 1. The candidate is a temporal learned anchor, not a stronger uniform floor

The only behavior intervention is on preflop continuation states after at least one non-forced preflop event:

`candidate = 0.75 * current_native_policy + 0.25 * lagged_native_policy`.

Root preflop and every postflop state return the current native H2 uncertainty-damped behavior unchanged. Heldout inference applies no anchor at all and evaluates the final learned AveragePolicy directly.

### 2. The lagged source has the intended iteration semantics

Before each Advantage refit, the implementation deep-copies the currently active four-member behavior ensemble. After the refit, the new ensemble becomes current and the saved pre-refit ensemble becomes the lagged source.

This yields the frozen sequence:

- iteration 1 traversal: current uniform, lagged uniform;
- iteration 2 traversal: current iteration-1 learned behavior, lagged still initial uniform;
- iteration 3 traversal: current iteration-2 learned behavior, lagged iteration-1 learned behavior.

Therefore the first intended causal divergence from Phase2B6 is iteration 3.

### 3. Exact Phase2B6 equivalence is actively checked before accepting the readout

The final evaluator compares candidate versus Phase2B6 on both seeds for iterations 1 and 2. It requires:

- `roots_added = 256` and exact equality with each Phase2B6 `i1c4` / `i2c4` stage report;
- exact Advantage-sample count;
- exact Strategy-sample count;
- exact intervention/root/postflop call counts;
- ensemble weighted NRMSE agreement within `1e-12`.

A mismatch aborts Phase2B8 rather than allowing a contaminated causal comparison.

### 4. Resume semantics preserve both current and lagged ensembles

The resume checkpoint stores separately:

- current four-member behavior model states;
- lagged four-member behavior model states;
- current and lagged uncertainty telemetry;
- anchor-policy telemetry;
- exact stage state and stage index.

The checkpoint is bound to the Phase2B8 execution SHA, H2/3H contract, action candidate and architecture fingerprint. The experiment can resume after interruption without recreating completed chunks or silently losing the lagged anchor state.

### 5. Control and heldout identities are independently hardened

Phase2B8 requires the exact Phase2B6 and Phase2B7 result hashes. It then reads Phase2B7's frozen inventory and rehashes:

- all four Phase2B6 control policy artifacts;
- both heldout gzip artifacts.

The representation, domain, training seeds, evaluation seeds and 1024-state policy slice are also checked before evaluation.

### 6. AveragePolicy fitting remains directly comparable

The candidate fits COMMON and NATIVE AveragePolicy models using the same capacity, audit seed convention, initialization/batch RNG conventions, policy-step budget, optimizer, batch size and LR used by the preceding controlled work. Policy artifacts carry a Phase2B8-specific schema and explicitly record `anchor_training=0.25` and `anchor_inference=0.0`.

Resume validation checks candidate identity, capacity, audit seed, anchor values and artifact SHA before reusing a policy fit.

### 7. Decision rule matches the precommit

The final decision requires all of the following for causal support:

- local Advantage gates pass;
- COMMON and NATIVE AveragePolicy fit gates pass;
- exact pre-divergence equivalence passes;
- both COMMON heldout means improve versus Phase2B6;
- pooled COMMON improvement is at least `0.015` absolute or `8%` relative;
- paired bootstrap 95% CI is strictly positive;
- no COMMON heldout p95 degrades by more than `0.02`;
- NATIVE pooled mean does not degrade by more than `0.01`;
- COMMON root mean does not worsen by more than `0.005` from the Phase2B6 root value;
- COMMON combined continuation mean does not worsen by more than `0.005` from the Phase2B6 continuation value.

Historical hard stability gates remain mean TV `<=0.15` and p95 TV `<=0.35` on both COMMON heldouts.

### 8. Synthetic tests cover the intervention algebra

The deterministic test file checks:

- root policy is unchanged;
- postflop policy is unchanged;
- an empty lagged source reproduces the exact Phase2B6 25% uniform algebra;
- a learned lagged source produces the frozen `75/25` policy mixture;
- output remains normalized/legal;
- region aggregation and frozen constants.

The tests are not claimed PASS on Ryzen yet. The launcher runs `py_compile` and the deterministic suite before building/running the real experiment.

### 9. Governance remains fail-closed

Phase2B8 is a small causal training screen, not production training. Even a hard stability PASS is only `STABILITY_ELIGIBLE_PENDING_STRENGTH`; it does not select an architecture. Strategic-strength comparison against the certified stable V1 control remains mandatory before architecture selection.

No higher uniform floor, alternate lag weight, seed shopping, threshold relaxation, production authorization or table deployment is allowed from this implementation.

## Audit conclusion

The implementation is consistent with the frozen Phase2B8 contract and is suitable for Ryzen execution. Its strongest validity feature is the explicit iteration-1/2 equivalence gate against Phase2B6, which should make the iteration-3 change interpretable as a narrow causal test of learned temporal anchoring rather than another broad retraining variation.

`READY FOR TABLES = NO`.
