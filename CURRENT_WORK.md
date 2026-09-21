# SpinCore Current Work

Date: 2026-09-21
Status: **PUBLIC SNAPSHOT RECONCILER PASS — OPENHOLDEM HEARTBEAT/LIFECYCLE TRACKER NEXT**

## Frozen strategy/runtime

Strategic candidate and all deployment identities remain frozen.

No training, EV tuning or holdout reuse is permitted.

## Public snapshot reconciler result

PASS over 12,000 public transitions.

Action coverage:
- FOLD 930;
- CHECK 702;
- CALL 4,734;
- BET_TO 859;
- RAISE_TO 3,451;
- ALL_IN 1,324.

Alias normalization:
- 805 all-in calls -> CALL;
- 3,505 stack-emptying BetTo/RaiseTo -> ALL_IN;
- alias failures 0.

Faults:
- 1,500 no-op snapshots rejected by one-action reconciler;
- 1,500 corrupt snapshots rejected;
- 1,196 skipped-action snapshots rejected.

The skipped-action gate did not record skipped attempts separately, so the next lifecycle test explicitly requires skipped_attempts == skipped_rejections.

## Active gate

A fail-closed heartbeat/lifecycle tracker is now implemented above the reconciler.

Contract:
- duplicate heartbeat = no-op;
- valid changed snapshot = exactly one canonical action appended;
- NewRound preserves transcript;
- MyTurn computes once per canonical state;
- repeated ProcessQuery returns cached decision only;
- changed state invalidates cache;
- wrong hand / corrupt / skipped transition latches failure;
- only HandReset clears the failure latch.

## Immediate action

```bash
bash tools/run_lt2_runtime_heartbeat_tracker.sh
```

Wait for `LT2_RUNTIME_HEARTBEAT_TRACKER_PASS`, then send
`SpinCore_LT2_runtime_heartbeat_tracker.json`.
