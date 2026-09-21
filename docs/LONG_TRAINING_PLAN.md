# SpinCore — Long-Training Plan

Status: **TRAINING CLOSED — WINDOWS SHADOW DLL MOCK PASS / REAL HOST PRELOAD**
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


## Windows shadow DLL mock-host result

PASS:
- exact Windows x64 user-DLL built;
- Windows core tests 2/2 PASS;
- LoadLibrary/mock-host PASS;
- host exports resolved;
- strict hand anchor and MyTurn inference PASS;
- probability/legal-mask checks PASS;
- repeated MyTurn cache PASS;
- hard-zero action-query barrier PASS.

Exact tested DLL SHA256:

`7566be1b3c73207d437171c2b4e94f6a94477786a2a48599994a647808030062`

## Next deployment gate

Inspect the architecture of the real `OpenHoldem.exe` before copying any DLL.

If the host is x64, the exact tested binary may proceed to the real shadow-load
gate. If the host is x86, build/test a separate x86 shadow DLL first.
