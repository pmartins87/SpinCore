# SpinCore Current Work

Date: 2026-09-21
Status: **LT3 CONTINUATION PERFORMANCE GATE — 8200 PRESERVED / OPENHOLDEM PAUSED**

## Strategic baseline

LT2 ENS8@8100 remains the frozen production baseline.

Its checkpoint, HU ENS8 sidecar, final holdout evidence and deployment artifacts
must remain read-only.

The previous OpenHoldem productionization work is preserved but **paused**.
No OpenHoldem host inspection, DLL installation or table testing is required
while LT3 training is the user's active priority.

## Primary active lane — LT3 continuation from preserved 8200

LT2 ENS8@8100 remains the frozen sealed-holdout-passed baseline.

The interrupted LT3 continuation has a durable matched checkpoint+sidecar at
iteration 8200.  It is preserved and remains eligible for continuation.

Why we are **not** restarting from zero:

- the Stage-B reservoir-poisoning hypothesis was not supported;
- controlled fresh refits showed usable signal remained in the mature reservoir;
- the material defect was insufficient HU fitting (fresh100), repaired by HU400;
- HU400 passed structural, broad and online-feedback validation;
- ENS8 then stabilized the mature HU current behavior and the 8100 candidate
  passed the pre-registered sealed holdout.

Thus there is no evidence that the accumulated 0..8100 learning state is
invalid.  A clean-from-zero run would be a separate research arm, not a required
repair.

Immediate engineering issue: the current ENS8 fit implementation is
unnecessarily sequential.  Before resuming 8201+, benchmark process-parallel
member fitting and require exact state/loss parity.


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

With the sequential continuation stopped and 8200 preserved, run:

```bash
bash tools/run_lt3_hu_ens8_parallel_fit_benchmark.sh
```

Expected sentinel:

`LT3_HU_ENS8_PARALLEL_FIT_BENCHMARK_PASS`

Do not restart H1 until this benchmark is adjudicated.
