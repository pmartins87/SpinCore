# SpinCore — LT2 public snapshot reconciler result

Date: 2026-09-21  
Status: **PASS — ONE PUBLIC TRANSITION CAN BE CANONICALIZED TO ONE EXACT LT2 ACTION**

## Scope

The gate exercised the runtime reconciler that observes only the public state before/after one voluntary action and infers the canonical LT2 transcript action.

No model inference, EV evaluation, optimizer work, training roots or holdout reuse occurred.

## Coverage

Total transitions:

- `12,000`.

By action:

- FOLD: `930`;
- CHECK: `702`;
- CALL: `4,734`;
- BET_TO: `859`;
- RAISE_TO: `3,451`;
- ALL_IN: `1,324`.

Domains:

- THREE_HANDED: `7,829`;
- TRUE_HEADS_UP: `4,171`.

Streets:

- preflop: `8,504`;
- flop: `2,597`;
- turn: `694`;
- river: `205`.

## Alias normalization

Validated canonicalization:

- all-in call -> CALL: `805` cases;
- BetTo/RaiseTo exactly to max stack -> ALL_IN: `3,505` cases.

Alias failures: **0**.

## Fault behavior

- duplicate/no-op snapshots rejected by the one-action reconciler: `1,500`;
- corrupted snapshots rejected: `1,500`;
- skipped-action snapshots rejected in `1,196` generated cases.

Primary transition failures: **0**.

### Important gate limitation

The original gate recorded only the number of skipped snapshots rejected, not the number of skipped snapshots attempted. Therefore it proves many skipped snapshots fail closed, but it does not by itself prove a 100% skipped-transition rejection rate.

The next lifecycle gate closes this accounting gap by explicitly recording both attempts and rejections and requiring equality.

## Decision

The canonical one-action inference semantics are accepted.

Next build the runtime heartbeat/lifecycle state machine that:

- treats duplicate heartbeats as no-ops;
- appends exactly one canonical action for a valid changed snapshot;
- preserves transcript across NewRound callbacks;
- caches one Hero decision per canonical state;
- returns only the cached value to repeated ProcessQuery calls;
- invalidates the cache when the state changes;
- latches any reconciliation/identity error until HandReset;
- proves skipped/corrupted/wrong-hand attempts all fail closed with exact attempt accounting.
