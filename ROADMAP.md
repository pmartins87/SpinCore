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
- OpenHoldem observable end-to-end tracker — **NEXT**;
- Windows native user-DLL — **AFTER E2E PASS**.

## Runtime pipeline now available

`OpenHoldem raw symbols -> strict adapter -> observable public snapshot -> canonical exact action transcript -> from-scratch authoritative rebuild -> frozen native inference -> exact lean action`

## Current gate

Prove this whole state-reconstruction path end-to-end without using any
solver-only fields that real OpenHoldem does not expose.

At Hero turns the rebuilt:
- SPNNIV1;
- SPNNIV2;
- actor/domain;
- active action mask;
- lean legal actions;
- exact action resolver

must match authoritative truth exactly.

After PASS, port the already-proven state machine and native inference core into
the Windows OpenHoldem user-DLL lifecycle.
