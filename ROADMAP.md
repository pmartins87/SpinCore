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
- native tracker + frozen inference shadow engine — **NEXT**;
- Windows OpenHoldem user-DLL binding — **AFTER SHADOW ENGINE PASS**;
- Windows DLL build + symbol/query dry-run — **AFTER BINDING**;
- log-only real-table shadow gate — **AFTER DLL DRY-RUN**;
- action-enabled table gate — **ONLY AFTER SHADOW PASS**.

## Native production chain

`OH raw frame -> native strict adapter -> native observable tracker -> canonical Hero state -> frozen native model -> sampled lean slot -> canonical exact action`

The next audit exercises the entire chain except actual OpenHoldem callbacks and actual table action.

## Safety / rollout order

No step may skip directly to action-enabled play.

Required sequence:
1. native shadow engine PASS;
2. Windows DLL compile/load PASS;
3. OpenHoldem callback/query dry-run PASS;
4. real-table log-only shadow PASS;
5. only then consider exposing action symbols.
