# SpinCore recovery status — 2026-08-09

## Physical evidence recovered

Recovered from the prior ChatGPT project conversation after it reached its size/file limit:

- `SpinCore_internal_current.bundle`
- `SpinCore_internal_current.tar.gz`
- `SpinCore_internal_continuity_manifest_R5.json`
- `SpinCore_R7_4_Ryzen_RunKit`

The recovered continuity manifest identifies:

- commit `445fb4a56da69f9a3ca48acbe13a6cfb95055ac7`
- tag `internal-r5-pass-reintegrated`
- `ready_for_tables = false`

The physical R5 checkout was rebuilt and revalidated on 2026-08-09:

- C++ regression: PASS
- Python R4/R5 regression: 12 PASS / 0 FAIL

## Historical certified state from the prior project conversation

The following state is historical conversation evidence and must not be confused with files physically present in the recovered R5 bundle:

- R0–R6: PASS
- R7.0: PASS
- R7.1: PASS
- R7.2: PASS
- R7.3: in progress; stability gate still FAIL

Last historical HU checkpoint:

- 5 iterations
- 128 roots/iteration
- 640 roots/seed
- seeds `20260807` and `20260829`
- cross-seed mean TV `0.3714`
- p50 TV `0.3471`
- p95 TV `0.6878`
- max TV `0.8978`

Approximation metrics at that checkpoint:

- seed 20260807: advantage weighted normalized RMSE `0.4529`; average-policy weighted mean TV `0.1128`
- seed 20260829: advantage weighted normalized RMSE `0.5462`; average-policy weighted mean TV `0.1122`

Frozen gates:

- Advantage weighted normalized RMSE <= `0.75`
- Average-policy weighted mean TV <= `0.12`
- Cross-seed mean TV <= `0.15`
- Cross-seed p95 TV <= `0.35`

Therefore `R7.3 = FAIL` at the last historical checkpoint and `READY FOR TABLES = NO`.

## Recovery invariants

1. Never relax a frozen gate to recover progress.
2. Distinguish recovered physical evidence from historical chat evidence.
3. No return to an old DeepSpin branch as if it were the current SpinCore.
4. Intermediate work stays internal; user-facing package delivery happens only at a meaningful final milestone or when explicitly requested.
5. Persist each meaningful recovery/roadmap step in this repository so a future chat can continue from Git instead of a transient conversation filesystem.
6. Final endpoint remains: **ready to start using at the tables**, only after implementation, production training, strategic audit, OpenHoldem integration, exploitation, and operational homologation.
