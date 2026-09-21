# SpinCore Roadmap — active state 2026-09-21

## Active status

- strategic candidate ENS8@8100 — **FROZEN / HOLDOUT PASS**;
- Python deployment parity — **PASS EXACT**;
- native C++ inference parity — **PASS**;
- hidden filler invariance — **PASS**;
- exact public-transcript rebuild — **PASS**;
- public snapshot -> canonical exact action reconciler — **PASS**;
- OpenHoldem heartbeat/lifecycle tracker + decision cache — **NEXT**;
- OpenHoldem symbol/scrape adapter — **AFTER TRACKER PASS**;
- Windows user-DLL integration — **AFTER ADAPTER**.

## Established runtime chain

`scraped public state -> normalized public snapshot -> canonical action reconciler -> exact transcript -> from-scratch solver rebuild -> frozen native inference -> exact lean action`

## Current gate

Validate lifecycle behavior around the reconciler:

- duplicate heartbeats;
- NewRound;
- MyTurn;
- repeated ProcessQuery;
- cache invalidation;
- hand identity;
- failure latch;
- HandReset recovery.

Skipped-transition fault injection now records exact attempts and exact rejections.

After PASS, bind the tracker to actual OpenHoldem state/symbol acquisition.
