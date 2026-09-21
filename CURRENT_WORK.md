# SpinCore Current Work

Date: 2026-09-21
Status: **LT3 H1 PERFORMANCE GATE — SEQUENTIAL LONG RUN SUPERSEDED / OPENHOLDEM PAUSED**

## Strategic baseline

LT2 ENS8@8100 remains the frozen production baseline.

Its checkpoint, HU ENS8 sidecar, final holdout evidence and deployment artifacts
must remain read-only.

The previous OpenHoldem productionization work is preserved but **paused**.
No OpenHoldem host inspection, DLL installation or table testing is required
while LT3 training is the user's active priority.

## Primary active lane — LT3 H1 performance gate

H1 is preregistered as a research-only continuation from the exact frozen
LT2 ENS8@8100 checkpoint+sidecar pair:

- source iteration: 8100;
- target iteration: 8600;
- additional iterations: 500;
- additional training roots: 300,000;
- 3H Advantage refit: fresh100;
- HU: ENS8, 8 members x fresh400;
- K4: off;
- workers: 31;
- Torch threads: 8;
- LT2 production artifacts: read-only;
- LT2 final holdout: retired / not reused;
- LT3 sealed holdout: untouched.

The initial sequential H1 launch is now considered **superseded for execution
quality**. Its strategy contract was valid, but the implementation did not
first benchmark the obvious independent parallelism across the eight HU members.

Before restarting H1, the new mandatory step is an exact-parity throughput
benchmark:

- reference: current sequential ENS8 fresh400 x8 fit;
- candidate: process-parallel ENS8 fitting;
- target host: Ryzen 9;
- exact member-state equality required;
- exact final-loss equality required;
- measured end-to-end speedup must include snapshot/serialization overhead.

Only after this gate passes may H1 restart from the frozen LT2@8100 source.

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

Stop any still-running sequential H1 process. Then run:

```bash
bash tools/run_lt3_hu_ens8_parallel_fit_benchmark.sh
```

Expected sentinel:

`LT3_HU_ENS8_PARALLEL_FIT_BENCHMARK_PASS`

Do not restart H1 until this benchmark is adjudicated.
