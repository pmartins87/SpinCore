# R7.5 Architecture Reset — Phase2B11 Implementation Audit

Status: **IMPLEMENTED / FROZEN BEFORE OUTPUTS / RYZEN TESTS NOT YET RUN**  
Date: 2026-08-24

## Audit scope

Reviewed the Phase2B11 factorized private/public chance estimator screen against the frozen Phase2B10 evidence, the explicit-deal solver contract, and the exact completed Phase2B6 behavior source.

Files under audit:

- `validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B11_FACTORIZED_CHANCE_ESTIMATOR_PRECOMMIT_20260824.md`
- `validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B10_RESULT_EVIDENCE_20260824.json`
- `tools/r7_5_arch_reset_v1plus_phase2b11_factorized_chance_estimator.py`
- `tools/test_r7_5_arch_reset_v1plus_phase2b11_factorized_chance_estimator.py`
- `tools/run_r7_5_arch_reset_v1plus_phase2b11_factorized_chance_estimator_ryzen.ps1`
- Phase2B10 explicit-deal implementation and tests.

Frozen Phase2B10 result SHA-256: `0295574c6133eb05866ecbdccf7e31efa4e6e8936dbd8bb7e375e166b27fe4dc`.

## Findings

### 1. Phase2B10 supports a genuinely mixed chance route

The uploaded result independently verifies:

- traversal-only TV `0.05516957953078942`;
- private-only TV `0.40876631080266784`;
- public-only TV `0.4470753497550347`;
- combined TV `0.5439893672482926`.

Private exceeds public in behavior seed `1342191342`, while public exceeds private in behavior seed `1801739323`. Therefore neither one-component dominance rule is satisfied. The mixed classification and factorized next route are the frozen consequence rather than a post-hoc preference.

### 2. The factorized estimator is marginally correct

For a fixed actor-hole pair, a private seed uniformly permutes the other 50 cards and takes the first four as the ordered opponent-hole assignment.

A public seed independently uniformly permutes the same 50 actor-excluded cards. For any fixed private assignment, removing those four cards from the public permutation leaves a uniformly random ordering of the valid 46-card board deck; taking its first five cards therefore gives a correct ordered board sample conditional on that private assignment.

Crossing private rows with shared public random-order columns changes dependence among samples, not the marginal distribution of any individual legal deal. The arithmetic mean of the crossed target cells is therefore an unbiased Monte-Carlo estimator of the same conditional root target expectation as the IID control.

### 3. Equal-compute controls are exact

Two budgets are compared:

- `FACTOR2X2` uses 4 traversals and is compared only to `IID4`, also 4 traversals;
- `FACTOR4X4` uses 16 traversals and is compared only to `IID16`, also 16 traversals.

No claim is based on comparing a higher-compute factorized estimator against a lower-compute IID control.

### 4. Raw targets are averaged at the estimator boundary

For every arm the individual root Advantage target vectors are generated first. The arm estimator is their simple arithmetic mean. Regret matching is then applied once for the diagnostic policy readout.

This is deliberately different from averaging already-discontinuous regret-matched policies and directly tests whether the chance estimator feeding Advantage learning can be stabilized.

### 5. Traversal noise is held out of the block comparison

The same deterministic traversal RNG seed is reused across chance draws and across the four estimator blocks for a given scenario/anchor. The private/public chance seeds change between blocks.

Thus paired block disagreement primarily measures chance-estimator instability rather than adding the already-small traversal-only component back into the test.

### 6. Every explicit deal remains fail-closed

Phase2B11 delegates target collection to the already-audited Phase2B10 explicit-deal path. Before accepting any target it requires exact equality of:

- acting-player root SPNNIV3 observation SHA;
- actor;
- universal legal tuple;
- legal mask.

The explicit-deal solver itself independently validates card ranges, uniqueness and live-seat assignment.

### 7. The exact best causally supported behavior source is preserved

Workers load the final four-member Advantage behavior states from each completed Phase2B6 checkpoint and use the exact Phase2B6 continuation intervention:

- root preflop native;
- preflop continuation 25% uniform floor;
- postflop native.

No Phase2B8 lagged policy or Phase2B9 Huber model enters this screen.

### 8. The primary gate is fixed before outputs

The primary comparison is `FACTOR4X4` versus `IID16`. PASS requires all of:

- material mean-TV improvement (`>=0.05` absolute or `>=20%` relative);
- material sign-disagreement improvement (`>=0.05` absolute or `>=15%` relative);
- directional TV improvement in both source behavior seeds;
- p95 TV no worse than IID16 by more than `0.02`;
- dominant-action mismatch no worse than IID16 by more than `0.02`;
- no factor-size reversal: FACTOR4X4 TV `<= FACTOR2X2 TV + 0.01`.

The 2×2 arm is secondary and cannot rescue a failed 4×4 primary gate post hoc.

### 9. Parallelism is appropriate

Each source behavior seed has 240 independent scenario/anchor/block tasks. Every task computes all four equal-compute designs locally. The launcher uses 30 independent worker processes × one Torch/OMP/MKL/OpenBLAS thread.

Unlike the earlier Phase2B9 fit screen, broad process-level parallelism here maps cleanly to independent solver work and should use the Ryzen efficiently without changing statistical semantics.

### 10. Workload and mutation firewall

Frozen total work:

`2 behaviors × 15 scenarios × 4 anchors × 4 blocks × 40 traversals = 19,200 root target traversals`.

There is no:

- model fit;
- optimizer step;
- Advantage or Strategy reservoir insertion;
- AveragePolicy fit;
- checkpoint mutation;
- architecture selection;
- production training.

### 11. Launcher revalidates the new solver path

Before scientific execution the launcher:

- checks exact B1/B6/B10 local result hashes against repository evidence;
- verifies Python/Torch/Numpy runtime identity;
- validates the H2/3H frozen model contract;
- rebuilds the x64 solver;
- requires ABI v2, SPNNIV3 and both explicit-deal diagnostic symbols;
- reruns the Phase2B10 explicit-deal round-trip tests;
- validates both exact Phase2B6 final behavior checkpoints.

### 12. Governance remains fail-closed

A Phase2B11 PASS permits only preparation of one separately precommitted small causal trajectory pilot with an equal-compute control. A FAIL routes either to regret-matching sensitivity analysis if raw/sign estimators improve coherently, or to solver-level chance-expectation/representation reassessment.

In every branch:

- no architecture winner is selected;
- R7.5.4/R8 remain blocked;
- production training remains unauthorized;
- `READY FOR TABLES = NO`.

## Audit conclusion

Phase2B11 is consistent with the frozen mixed-private/public Phase2B10 evidence and tests the next causal question at equal compute. The crossed sampling construction preserves each deal's correct conditional marginal while deliberately changing sample coupling. The implementation is suitable for Ryzen execution **subject to the launcher's real py_compile, synthetic tests, CMake/MSVC build, explicit-deal round-trip tests, and checkpoint preflights passing**.
