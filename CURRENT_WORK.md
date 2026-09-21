# SpinCore Current Work

Date: 2026-09-21
Status: **LT3 HEAVY H1 TRAINING — PRIMARY ACTIVE LANE / OPENHOLDEM PAUSED**

## Strategic baseline

LT2 ENS8@8100 remains the frozen production baseline.

Its checkpoint, HU ENS8 sidecar, final holdout evidence and deployment artifacts
must remain read-only.

The previous OpenHoldem productionization work is preserved but **paused**.
No OpenHoldem host inspection, DLL installation or table testing is required
while LT3 training is the user's active priority.

## Primary active lane — LT3 Heavy H1

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

The H1 run has a hard stop at iteration 8600.

After H1 completes, the next step is **development-set adjudication**. Do not
extend automatically to H2 and do not touch any sealed holdout.

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

Run:

```bash
bash tools/run_lt3_heavy_ens8_h1.sh
```

Expected completion sentinel:

`LT3_HEAVY_ENS8_H1_TRAINING_PASS`

Then send:

`SpinCore_LT3_heavy_ens8_H1.json`

Do not restart or extend past 8600 before H1 adjudication.
