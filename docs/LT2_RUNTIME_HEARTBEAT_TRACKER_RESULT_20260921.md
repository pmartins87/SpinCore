# SpinCore — LT2 OpenHoldem heartbeat/lifecycle tracker result

Date: 2026-09-21  
Status: **PASS — FAIL-CLOSED LIFECYCLE/CACHE STATE MACHINE VALIDATED**

## Scope

The gate validated the runtime tracker above the public-snapshot reconciler.

No deployment-model inference, EV evaluation, optimizer work, training roots or holdout reuse occurred.

## Coverage

- canonical public transitions: `10,000`;
- action events appended: `10,000`;
- duplicate heartbeats: `20,000`;
- MyTurn callbacks: `7,862`;
- actual decision computations: `3,931`;
- repeated MyTurn cache hits: `3,931`;
- ProcessQuery cache reads: `11,793`;
- NewRound callbacks: `1,170`;
- cache invalidation checks: `3,931`.

## Exactness

- action mismatches: **0**;
- transcript mismatches: **0**;
- cache invalidation failures: **0**;
- lifecycle failures: **0**.

Every canonical Hero state was computed exactly once despite repeated MyTurn callbacks.

Every tested ProcessQuery returned the cached decision only.

## Fault behavior

Corrupted snapshot:
- attempts `500`;
- rejected `500`.

Wrong hand identity:
- attempts `500`;
- rejected `500`.

Skipped transitions:
- generated attempts `493`;
- rejected `493`.

Failure latch:
- checks `500`;
- pass `500`.

HandReset recovery:
- attempts `500`;
- pass `500`.

## Decision

The lifecycle/cache contract is accepted:

- duplicate heartbeat = no-op;
- one public transition = exactly one canonical transcript action;
- NewRound preserves transcript;
- MyTurn computes once per canonical state;
- ProcessQuery is cache-only;
- state change invalidates cache;
- synchronization/identity failure latches until HandReset.

The remaining integration boundary is the actual OpenHoldem scrape/symbol layer.

The next gate validates the strict mapping from documented OpenHoldem values
(`balanceN`, `currentbetN`, `pot`, dealt/playing/all-in bits, blinds,
dealer/user chairs, betround and cards) into the normalized LT2 hand anchor and
observable public snapshot.
