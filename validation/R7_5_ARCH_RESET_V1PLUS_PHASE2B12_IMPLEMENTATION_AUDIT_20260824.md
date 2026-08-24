# R7.5 Architecture Reset — Phase2B12 Implementation Audit

Status: **IMPLEMENTED / FROZEN BEFORE OUTPUTS / RYZEN TESTS NOT YET RUN**  
Date: 2026-08-24

## Audit scope

Reviewed Phase2B12 against the frozen Phase2B11 failure and the Phase2B12 precommit.

Files under audit:

- `validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B12_IID_CHANCE_EXPECTATION_CONVERGENCE_PRECOMMIT_20260824.md`
- `validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B11_RESULT_EVIDENCE_20260824.json`
- `tools/r7_5_arch_reset_v1plus_phase2b12_iid_chance_expectation_convergence.py`
- `tools/test_r7_5_arch_reset_v1plus_phase2b12_iid_chance_expectation_convergence.py`
- `tools/run_r7_5_arch_reset_v1plus_phase2b12_iid_chance_expectation_convergence_ryzen.ps1`
- inherited Phase2B10 explicit-deal solver extension and tests;
- inherited Phase2B11 legal IID conditional-deal generator.

Frozen evidence:

- Phase2B1 SHA-256 `f95751afeb17fcd5844bfcb2971577b92a400750444e5dabe2f4ddb5718ba6ef`;
- Phase2B6 SHA-256 `33ec6ba89823dae632b7af935def17444379c96a28e59478c0b7c91f1ec3659a`;
- Phase2B10 SHA-256 `0295574c6133eb05866ecbdccf7e31efa4e6e8936dbd8bb7e375e166b27fe4dc`;
- Phase2B11 SHA-256 `1596023d39609ddfe5a6528a2e62d376c8e6bd29dde68d24a20a9b0ed782b1aa`.

## Findings

### 1. The next question follows directly from the B11 failure

Phase2B11 rejected private/public crossing at equal compute, but ordinary IID averaging showed a strong size effect: IID16 mean TV `0.33467` versus IID4 `0.52304`. B12 therefore does not retry factorization or tune a failed candidate. It isolates the surviving hypothesis: plain IID conditional chance expectation may converge with larger K.

### 2. K values are nested, not independently seed-shopped

Every block generates one deterministic 64-deal IID stream. K8, K16, K32 and K64 are prefix means of that same stream. Thus apparent K improvement cannot be manufactured by giving each K a favorable independent seed family.

### 3. K16 is an exact historical reproduction control

B12 calls the exact Phase2B11 IID generator with namespace `301`, the same scenario, anchor and block identifiers, and the same fixed traversal seed. Therefore samples `0..15` are exactly the B11 IID16 sample set.

Before any K32/K64 decision is accepted, B12 requires within `1e-12` exact reproduction of six B11 IID16 summaries: pooled mean TV, sign disagreement, target MAD, dominant mismatch and both per-source-seed mean TVs. A mismatch raises an exception and prevents scientific classification.

### 4. Conditional chance distribution remains legal

The acting player's exact hole cards stay fixed. Opponent private cards and future board are generated through the already-audited Phase2B11 IID conditional-deal generator. Every explicit deal is valid and the inherited Phase2B10 `_one_target` requires exact root SPNNIV3 bytes, actor and legal identity before accepting a target.

### 5. Traversal noise is held out of the convergence curve

Each scenario/anchor uses the frozen Phase2B11 traversal seed for all 64 chance samples and all four blocks. The varying dimension is hidden/future chance, matching the causal question established by Phase2B1/B10.

### 6. Raw targets are averaged before regret matching

For each K, the ten-slot raw Advantage target vector is averaged first and regret matching is applied only for diagnostic pair comparison. This is a target-expectation diagnostic, not the rejected Phase2B0 behavior-algebra intervention.

### 7. The metric set explicitly tracks the heavy tail

B11 p95 TV saturated at `1.0`, which makes p95 alone uninformative for moderate convergence. B12 therefore additionally reports `tail_rate_tv_ge_035`, the fraction of paired estimators whose diagnostic regret-matching TV is at least the historical `0.35` p95 stability threshold. The precommit requires material tail-rate reduction for a full PASS.

### 8. Gates are frozen and deliberately stronger than mere directional improvement

A full K64 screen PASS requires:

- at least `0.08` absolute **and** `25%` relative mean-TV reduction versus reproduced K16;
- K64 pooled mean TV `<= 0.24`;
- material sign-disagreement improvement;
- material `TV >= 0.35` tail-rate improvement;
- improvement for both source behavior seeds;
- monotone K16 -> K32 -> K64 convergence within `0.01` tolerance;
- no dominant-action mismatch degradation above `0.02`.

A weaker `>=0.05` absolute mean-TV gain for both seeds is classified only as slow convergence and authorizes no training.

### 9. Work is parallelized at the natural task level

The workload is:

`2 behaviors × 15 scenarios × 4 anchors × 4 blocks × 64 samples = 30,720 root traversals`.

The launcher uses 30 independent worker processes with one Torch/OMP/MKL/OpenBLAS thread each. Worker count changes compute scheduling only; all random seeds and pairings are explicit and deterministic.

### 10. No learned state is mutated

B12 performs no optimizer step, model fit, reservoir insertion, Strategy collection, AveragePolicy fit or checkpoint mutation. Phase2B6 checkpoints are loaded only to reconstruct the frozen four-member behavior ensembles.

### 11. Governance remains fail-closed

- factorized chance estimator remains rejected;
- no higher preflop floor is authorized;
- no lag-weight or Huber tuning is reopened;
- no result-dependent K selection is allowed;
- a B12 full PASS allows only a separately precommitted small causal training pilot with an equal-compute control;
- SLOW/PLATEAU authorizes no training;
- no architecture winner is selected;
- R7.5.4/R8 remain blocked;
- `READY FOR TABLES = NO`.

## Audit conclusion

The Phase2B12 implementation matches the frozen scientific question and is suitable for Ryzen execution **subject to the launcher's real Python tests, CMake/MSVC build, explicit-deal round-trip tests, prerequisite SHA checks and mandatory B11 IID16 reproduction gate passing**.

This phase should settle whether ordinary solver-level chance integration has a practically useful convergence trajectory before any further training intervention is considered.
