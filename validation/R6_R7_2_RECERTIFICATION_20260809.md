# SpinCore R6 / R7.0-R7.2 physical recertification — 2026-08-09

> **SUPERSEDED FOR CURRENT-CHECKOUT REPRODUCIBILITY by `validation/RECOVERY_INTEGRITY_AUDIT_20260809.md`.**
>
> This document preserves the results and hashes recorded during the prior transient recovery run. A later audit found that the full source/test tree used for those tests was not persisted into reachable GitHub `main`. Therefore the numbers below remain recovery evidence, but they are **not** a claim that the present checkout can reproduce them until the missing tree is restored and rerun.

## Result recorded by the transient recovery run

- R6: PASS recertified
- R7.0: PASS recertified
- R7.1: PASS recertified
- R7.2: PASS recertified
- R7.3: still FAIL / in progress
- READY FOR TABLES: NO

## Physical validation recorded at that time

- C++ Release: 76 PASS / 0 FAIL
- C++ ASan + UBSan: 76 PASS / 0 FAIL
- Python: 47 PASS / 0 FAIL
- C ABI: SPINCORE_SOLVER_C_ABI_V2

## Recovered/recertified semantics recorded at that time

- Exact cloneable `SpinTraversalState` preserves hidden cards and future board across action branches.
- Production terminal utility is explicit-payout exact ICM `V(after)-V(before)`; WTA reduces exactly to chip fraction.
- True HU preserves the already locked third place.
- Simultaneous elimination with unequal unresolved lower-place payouts fails closed.
- External-sampling advantage targets are `V(I,a) - sum sigma(a)V(I,a)`.
- Average policy uses own-reach semantics.
- R7.1 C++ frontier enumerates non-target branches until target actor or terminal, with node/depth hard caps.
- Native frontier is independently checked against a second C++ enumeration and against the Python own-reach collector.
- R7.2 audit selection spans the full reservoir deterministically instead of using an oldest-prefix slice.
- Approximation reporting separates unweighted and LCFR-weighted metrics; gates use weighted metrics.
- R7 checkpoint state includes mid-iteration phase/root/optimizer progress and all relevant RNG states.
- Continuous run equals stop/save/destroy/restore/continue in the deterministic recovery test.
- Bounded fresh-process optimizer worker reloads and atomically rewrites the checkpoint.

## Frozen R7.3 evidence

Historical last checkpoint: 640 roots/seed; mean TV 0.3714; p95 TV 0.6878. Frozen gates remain mean <= 0.15 and p95 <= 0.35.

## File SHA-256 recorded by the transient run

- `CMakeLists.txt`: `163abf38ebcbf5d1cf961febd3bc2a28798b9d24c7f1f67ac9bd3df3a3822b4c`
- `include/spincore/solver_c_api.h`: `c741d61c9465cf5984645d48b9b1f4c3d3a072b348935c4c01ec237f9fc057c3`
- `src/solver_c_api.cpp`: `397345ce278998c2aa59ecd2a3045dc664a1779311ac44ab84a3ed668de56295`
- `python/spincore/solver.py`: `a60c13533ea42409091f4e953f75a9bce3dfe7769a0e1a83d0caa80699782658`
- `python/spincore/deep_cfr.py`: `aadd951db6954eb83f675c89bec0fc9b21a4b4ba79a78436941f4a092ffa76fd`
- `python/spincore/r7.py`: `6ad601fccd581dab35924f0c472c98262379eda10fbbe9e54a6167682ff69226`
- `tools/r7_training_worker.py`: `8de9b66a63984135d2cdebf5da4c4d936648f5bb39ed7178fa85e0bb17d33dd1`
- `tests/test_solver_frontier.cpp`: `e98ea933ce1655072e51cba9db89ecdcb255834201b80fa3ddc710c02f93d2f4`
- `python_tests/test_r7.py`: `bbb95236c82c5591458966f22a00e6abfb802073bbf63f02eba9a1da87b2a79c`
