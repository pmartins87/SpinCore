# SpinCore — LT2 Windows OpenHoldem shadow DLL mock-host result

Date: 2026-09-21  
Status: **PASS — WINDOWS USER-DLL ABI / MOCK HOST VALIDATED**

## Scope

The exact Windows x64 `SpinCore_LT2_Shadow.dll` was:

1. built with MSVC 19.44 / Visual Studio 2022;
2. linked against the native SpinCore core;
3. loaded with Windows `LoadLibrary` by a mock OpenHoldem host;
4. connected to mock `GetSymbol`, `GetHandnumber`, and `WriteLog` exports;
5. driven through the real user-DLL callbacks;
6. kept in hard shadow-only mode.

No real OpenHoldem process and no table action were involved.

## Build/test evidence

- Windows frozen bundle SHA256: PASS;
- Windows DLL build: PASS;
- core CTest suite: 2/2 PASS;
- mock-host verdict: PASS.

Mock decision:
- host loaded: 1;
- hand active: 1;
- decision ready: 1;
- failure latch: 0;
- domain: TRUE_HEADS_UP;
- sampled slot: 3;
- exact action type: 4 (RaiseTo);
- exact amount_to: 40;
- probability sum: 1;
- legal action count: 4;
- repeated MyTurn cache stable: 1;
- action-query barrier: 1.

## Tested Windows DLL identity

`SpinCore_LT2_Shadow.dll`

SHA256:

`7566be1b3c73207d437171c2b4e94f6a94477786a2a48599994a647808030062`

This exact binary is the one that must be used for the first real OpenHoldem
shadow-load gate.

## Safety result

The mock host confirmed the action-control query barrier remains hard-zero.

The DLL can compute/log the selected action but cannot authorize an OpenHoldem
table action in this build.

## Decision

The Windows ABI/mock-host boundary is accepted.

Before copying the tested DLL into a real OpenHoldem installation, inspect the
actual `OpenHoldem.exe` PE architecture.

The current tested DLL is x64. If the real OpenHoldem executable is x86, a
separate x86 build/mock-host gate is required. Do not attempt to load an
architecture-mismatched DLL.
