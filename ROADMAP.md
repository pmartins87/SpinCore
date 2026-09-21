# SpinCore Roadmap — active state 2026-09-21

## Active status

- strategic candidate ENS8@8100 — **FROZEN / HOLDOUT PASS**;
- Python deployment parity — **PASS EXACT**;
- native C++ inference parity — **PASS**;
- hidden filler invariance — **PASS**;
- exact public-transcript rebuild — **PASS**;
- public snapshot -> exact action reconciler — **PASS**;
- heartbeat/lifecycle tracker + cache — **PASS**;
- OpenHoldem symbol/scrape adapter — **PASS**;
- observable OpenHoldem end-to-end tracker — **PASS**;
- native C++ OpenHoldem observable tracker — **PASS**;
- native tracker + frozen inference shadow engine — **PASS**;
- Windows user-DLL compile/load + mock-host gate — **PASS**;
- real OpenHoldem host architecture inspection — **NEXT**;
- real OpenHoldem shadow load — **AFTER HOST INSPECTION PASS**;
- real-table log-only shadow gate — **AFTER OPENHOLDEM LOAD PASS**;
- action-enabled gate — **ONLY AFTER REAL SHADOW PASS**.

## Windows user-DLL gate

The Windows ABI/mock-host gate is now PASS.

A mock host emulates the minimum OpenHoldem exports required by the DLL and
loads the produced `SpinCore_LT2_Shadow.dll`.

Required properties:
1. all core/unit tests pass under MSVC;
2. DLL loads via Windows `LoadLibrary`;
3. host function lookup succeeds;
4. internal bundle SHA256 gate succeeds;
5. a strict hand anchor is built from OH symbols;
6. MyTurn produces exactly one cached shadow decision;
7. probability mass/legal mask are valid;
8. all action-control dll$ queries remain zero;
9. no table action path exists in this build.

After PASS, load the exact same DLL into OpenHoldem with autoplayer/action use
still disabled and inspect shadow logs.


## Parallel LT3 research lane

- LT3 H1 plan preregistered — **READY**;
- LT3 H1 8100->8600 heavy continuation — **NEXT / PARALLEL**;
- LT3 development battery — **AFTER H1**;
- LT3 H2 — **CONDITIONAL ON H1**;
- LT3 sealed holdout — **ONLY AFTER ALL RESEARCH CHOICES ARE FROZEN**;
- LT3 deployment promotion — **NOT AUTHORIZED**.

LT3 cannot modify the frozen LT2 production artifacts or reuse the LT2 final
holdout.


## Real OpenHoldem pre-load inspection

The exact mock-tested DLL is x64 and has SHA256:

`7566be1b3c73207d437171c2b4e94f6a94477786a2a48599994a647808030062`

Before installation:
1. identify the actual `OpenHoldem.exe` used by the user;
2. parse its PE machine architecture;
3. inspect any existing `user.dll`;
4. verify the exact tested shadow DLL hash;
5. require host/DLL architecture compatibility.

This inspection is read-only and does not modify the OpenHoldem directory.
