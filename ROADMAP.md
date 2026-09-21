# SpinCore Roadmap — active state 2026-09-21

## Active status

- strategic candidate ENS8@8100 — **FROZEN / HOLDOUT PASS**;
- Python deployment parity — **PASS EXACT**;
- native C++ inference parity — **PASS**;
- hidden filler invariance — **PASS**;
- exact public-transcript rebuild — **PASS**;
- public snapshot -> canonical exact action reconciler — **PASS**;
- heartbeat/lifecycle tracker + decision cache — **PASS**;
- OpenHoldem symbol/scrape adapter — **NEXT**;
- observable-snapshot reconciler integration — **AFTER ADAPTER PASS**;
- Windows user-DLL bridge — **AFTER OBSERVABLE INTEGRATION**.

## Proven lifecycle behavior

The tracker survives duplicate heartbeats and NewRound callbacks without
duplicating transcript state, computes a Hero decision exactly once per
canonical state, serves repeated ProcessQuery calls from cache, invalidates that
cache on state mutation and rejects every generated corrupted/wrong-hand/skipped
transition.

## Current boundary

The next risk is not solver or strategy logic. It is translating raw OpenHoldem
symbols and physical 0..9 chairs into exact LT2 chip/card/seat semantics.

A strict adapter is now implemented and tested by round-tripping authoritative
solver states through synthetic OpenHoldem frames plus malformed-frame fault
injection.

After PASS, the reconciler will consume the reduced observable OpenHoldem
snapshot rather than a solver-generated full PublicSnapshot.
