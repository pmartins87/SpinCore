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
- native C++ OpenHoldem observable tracker — **NEXT**;
- Windows OpenHoldem user-DLL binding — **AFTER NATIVE TRACKER PASS**;
- log-only/shadow table gate — **AFTER DLL BUILD**;
- action-enabled table gate — **ONLY AFTER SHADOW PASS**.

## Proven reference pipeline

`OpenHoldem raw symbols -> strict adapter -> silent-check-aware reconciliation -> exact public transcript -> from-scratch authoritative rebuild -> canonical Hero state`

The full reference path passed with zero action, transcript or canonical-state mismatches.

## Productionization step

The same state semantics are now being moved into native C++ so the final
OpenHoldem DLL has no Python runtime dependency.

The native tracker audit is mechanical only and does not touch strategy, model
weights, EV or holdout evidence.

After PASS:
1. combine native tracker + already-validated native neural runtime;
2. bind actual OpenHoldem callbacks/GetSymbol access;
3. verify bundle identity/hash fail-closed;
4. compile Windows user-DLL;
5. run log-only shadow mode before any action is enabled.
