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
- Windows user-DLL compile/load + mock-host gate — **NEXT**;
- real OpenHoldem shadow load — **AFTER WINDOWS MOCK PASS**;
- real-table log-only shadow gate — **AFTER OPENHOLDEM LOAD PASS**;
- action-enabled gate — **ONLY AFTER REAL SHADOW PASS**.

## Windows user-DLL gate

The next gate validates the actual Windows ABI boundary without touching a real
table.

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
