# SpinCore — LT2 OpenHoldem observable tracker E2E result

Date: 2026-09-21  
Status: **PASS — RAW OPENHOLDEM OBSERVATION TO CANONICAL HERO STATE VALIDATED**

## Scope

The gate exercised the complete observable-state pipeline:

`synthetic OpenHoldem raw frame -> strict symbol adapter -> observable reconciliation -> canonical exact transcript -> from-scratch solver rebuild -> Hero canonical parity`

No deployment-model inference, EV evaluation, optimizer work, training roots or holdout reuse occurred.

## Coverage

- public transitions: `10,000`;
- duplicate heartbeats: `10,000`;
- Hero canonical state checks: `3,916`;
- actual street reveals: `1,972`;
- silent CHECK deferrals: `1,769`;
- delayed actions reconciled at MyTurn: `719`;
- multi-action synchronization events: `689`.

Hero domain coverage:
- THREE_HANDED: `2,143`;
- TRUE_HEADS_UP: `1,773`.

Hero street coverage:
- preflop: `1,710`;
- flop: `1,095`;
- turn: `682`;
- river: `429`.

## Exactness

- exact-action mismatches: **0**;
- canonical-state mismatches: **0**;
- transcript mismatches: **0**;
- gate failures: **0**.

## Fault rejection

- corrupt observable frames: `500/500` rejected;
- skipped observable transitions: `500/500` rejected.

## Important semantics now proven

1. OpenHoldem-invisible CHECKs can safely remain pending.
2. `DLLUpdateOnMyTurn` can be used as synchronization evidence that actor order reached Hero.
3. Later observable changes can reconcile one or more silent CHECKs plus one visible action.
4. Real flop/turn/river reveals can replace earlier hidden filler cards because the canonical state is rebuilt from scratch.
5. OpenHoldem card-derived betround semantics correctly handle all-in runouts where all five cards become visible while the internal solver betting street remains earlier.
6. The rebuilt Hero SPNNIV1/SPNNIV2 and lean action semantics remain exact.

## Decision

The Python/reference OpenHoldem integration architecture is accepted.

The next gate ports the observable adapter/tracker/rebuild semantics into native C++ — the code path intended for the final Windows `user.dll` — and audits it independently before binding Windows/OpenHoldem callbacks and the frozen native model.
