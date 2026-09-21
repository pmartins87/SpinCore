# SpinCore Current Work

Date: 2026-09-21
Status: **LT3 CLEAN-REBUILD DESIGN — CONTINUATION SUPERSEDED / OPENHOLDEM PAUSED**

## Strategic baseline

LT2 ENS8@8100 remains the frozen production baseline.

Its checkpoint, HU ENS8 sidecar, final holdout evidence and deployment artifacts
must remain read-only.

The previous OpenHoldem productionization work is preserved but **paused**.
No OpenHoldem host inspection, DLL installation or table testing is required
while LT3 training is the user's active priority.

## Primary active lane — LT3 clean rebuild

LT2 ENS8@8100 remains preserved as the strongest validated baseline, but it
is no longer the starting point for the next research training line.

Reason:

- the serious LT1/LT2 lineage through iteration 7500 used HU fresh100;
- later causal work showed fresh100 was insufficient on the mature HU reservoir;
- iterations 7501..8000 repaired this with HU fresh400;
- iterations 8001..8100 added online ENS8 fresh400;
- the 8100 candidate passed its frozen holdout, so this history does not make
  8100 invalid;
- nevertheless, it is a mixed-lineage candidate rather than a clean run of the
  corrected training schedule from iteration 0.

The interrupted continuation reached a durable checkpoint at iteration 8200.
Preserve it as historical evidence only.  Do not resume it.

Before a fresh LT3 run, benchmark independent HU ensemble fitting on the Ryzen
and freeze the fastest exact-parity execution layout.  Then freeze the
from-zero algorithmic schedule before generating new roots.


## OpenHoldem deployment lane — PAUSED

Completed before pause:

- observable E2E tracker PASS;
- native C++ tracker PASS;
- native tracker + frozen inference shadow engine PASS;
- Windows x64 shadow user-DLL build PASS;
- Windows mock-host LoadLibrary/ABI gate PASS.

Paused next step:

- inspect the actual OpenHoldem host architecture.

No deployment work is needed now.

## Immediate action

With the sequential continuation stopped, run:

```bash
bash tools/run_lt3_hu_ens8_parallel_fit_benchmark.sh
```

Expected sentinel:

`LT3_HU_ENS8_PARALLEL_FIT_BENCHMARK_PASS`

Do not restart H1 until this benchmark is adjudicated.
