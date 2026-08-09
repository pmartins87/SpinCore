# SpinCore recovery integrity audit — 2026-08-09

## Finding

The repository state reached after the interrupted recovery conversation was internally inconsistent. Documentation said R6/R7.0–R7.2 had been physically recertified, but the reachable `main` tree could not reproduce that claim.

### Physical contradictions found

1. `CMakeLists.txt` references the complete poker engine (`betting_engine.cpp`, `hand_engine.cpp`, `tournament_value.cpp`, neural/core files, tests, etc.), but many of those files are absent from the reachable tree.
2. `validation/R6_R7_2_RECERTIFICATION_20260809.md` records hashes for `python/spincore/r7.py`, `tools/r7_training_worker.py`, `tests/test_solver_frontier.cpp`, and `python_tests/test_r7.py`; those files are absent from current `main`.
3. The persisted Python solver wrapper used pre-V2 function names while `include/spincore/solver_c_api.h` and `src/solver_c_api.cpp` expose the canonical `SPINCORE_SOLVER_C_ABI_V2` names.
4. The recovered R5 commit `445fb4a56da69f9a3ca48acbe13a6cfb95055ac7` and historical pre-loss commit `f7eeb4463c056252bf409fcdf00cd2af5bb821be` are referenced by recovery metadata but are not fetchable as commits from the current GitHub repository.
5. File Library searches for `SpinCore_internal_current.bundle`, `SpinCore_internal_current.tar.gz`, the R5 continuity manifest, and the missing R7 source names did not locate the recovery artifacts.

## What remains trustworthy

- The historical/recovery *semantic decisions* and frozen gates remain authoritative unless contradicted by later evidence.
- The historical R7.3 result remains FAIL: 640 roots/seed, mean TV 0.3714, p95 TV 0.6878.
- `READY FOR TABLES = NO`.
- No gate is relaxed by this correction.

## Repair performed in this commit

- `python/spincore/solver.py` is realigned to the persisted ABI V2 header: canonical creation/apply/neural/ICM/frontier symbols, explicit dead-player sequence, and fail-closed error handling.
- `STATUS.json` and `ROADMAP.md` no longer describe the current checkout as physically recertified.
- The previous validation note is explicitly superseded for *reproducibility status*. Its historical metrics and recorded hashes remain useful recovery evidence.

## Blocking next step

Restore the missing self-contained R0–R7.2 source/test tree from an authoritative recoverable artifact or source. Only after a clean build and full regression should R7.3 training resume.
