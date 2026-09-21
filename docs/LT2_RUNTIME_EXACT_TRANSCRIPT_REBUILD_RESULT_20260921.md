# SpinCore — LT2 exact public-transcript rebuild result

Date: 2026-09-21  
Status: **PASS — CANONICAL STATE CAN BE REBUILT FROM HERO/VISIBLE BOARD + EXACT PUBLIC TRANSCRIPT**

## Scope

At each synthetic Hero decision, the original live solver state was discarded.

The state was rebuilt from:

- original tournament scenario;
- Hero hole cards;
- currently visible board;
- deterministic legal fillers for hidden cards;
- exact public voluntary action transcript.

The transcript was replayed through:

`spincore_solver_state_apply_exact`

No inference, EV, optimizer work, training roots or holdout reuse occurred.

## Coverage

- target Hero states: `5,000`;
- alternate from-scratch rebuilds: `20,000`;
- legal exact-action resolutions compared: `83,724`.

Domains:

- THREE_HANDED: `2,856`;
- TRUE_HEADS_UP: `2,144`.

Streets:

- preflop: `1,487`;
- flop: `1,367`;
- turn: `1,162`;
- river: `984`.

Transcript path length:

- min: `0`;
- max: `15`;
- mean: `4.7894`.

## Result

Failures: **0**.

Every rebuild reproduced exactly:

- actor/domain;
- SPNNIV1;
- SPNNIV2;
- active mask;
- lean legal actions;
- exact resolution of every current lean action;
- visible-board count.

## Decision

The canonical solver state does not need to persist through the whole hand.

The production OpenHoldem bridge may rebuild from scratch at Hero decisions using the current visible cards and an accumulated exact public transcript.

The remaining live-integration problem is now transcript acquisition:

**successive public table snapshots -> one canonical exact action -> accumulated transcript**

This must be deterministic and fail closed on skipped/corrupted/ambiguous snapshots.
