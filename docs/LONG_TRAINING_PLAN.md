# SpinCore — Long-Training Plan

Status: **TRAINING CLOSED — WINDOWS OPENHOLDEM SHADOW DLL**
Date: 2026-09-21

## Completed gates

- strategic holdout PASS;
- Python deployment parity PASS;
- native C++ inference parity PASS;
- hidden filler invariance PASS;
- exact transcript rebuild PASS;
- public action reconciliation PASS;
- lifecycle/cache PASS;
- OpenHoldem symbol adapter PASS;
- observable E2E PASS;
- native C++ tracker PASS;
- native tracker + frozen inference shadow engine PASS.

## Native shadow decision evidence

The complete native decision core passed:
- 1,600 decisions;
- 1,600 inference calls;
- 1,600 repeated MyTurn cache hits;
- 1,600 exact selected-action resolutions;
- both strategy domains;
- all streets;
- 0 failures.

## Windows production boundary

A real shadow-only user-DLL is now implemented.

Important safeguards:
- Windows callback ABI matches OpenHoldem's user-DLL interface;
- OpenHoldem host functions are resolved dynamically from the executable;
- frozen model file SHA256 is verified inside the DLL;
- embedded model/checkpoint/ensemble identities remain validated;
- card conversion uses OpenHoldem suit constants and SpinCore's current
  rank-major id convention;
- invalid/missing symbols fail closed after a hand anchor is active;
- no action query returns a positive authorization in this build.

## Current gate

Build the Windows x64 DLL with MSVC, run SpinCore unit tests, then load the DLL
into a mock OpenHoldem host process.

Only after this passes should the DLL be introduced into OpenHoldem itself.
