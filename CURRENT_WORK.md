# SpinCore Current Work

Date: 2026-09-21
Status: **OPENHOLDEM OBSERVABLE E2E FIRST RUN DIAGNOSED — INVISIBLE CHECK SYNCHRONIZATION PATCHED**

## Frozen strategy/runtime

All strategic/model identities remain frozen.

No training, EV tuning or holdout reuse is permitted.

## First observable-E2E failure

The first run stopped after only 7 transitions with:

- exact action mismatches: 0;
- canonical state mismatches: 0;
- transcript mismatches: 0;
- failure kind: `transition`;
- failure reason: null.

That pattern identified an observable no-change, not a solver/model divergence.

## Root cause

A poker CHECK changes none of the OpenHoldem fields used by the strict
observable projection:

- balances;
- current bets;
- pot;
- fold/all-in bits;
- board cards.

Therefore an opponent CHECK can be real while the OpenHoldem public snapshot is
byte-for-byte unchanged.

The previous observable tracker incorrectly assumed that every voluntary action
must immediately produce a changed snapshot and therefore returned NO_CHANGE
where the E2E harness demanded ACTION.

This is a runtime-observability issue, not a strategic issue.

## Corrected contract

The observable tracker now:

1. treats an unchanged heartbeat as a no-op;
2. permits pending silent CHECKs;
3. when later observable evidence appears, deterministically advances through
   required CHECKs before the one chip/fold-changing action;
4. permits at most one chip/fold-changing action per scrape interval;
5. on `DLLUpdateOnMyTurn`, uses the callback itself as evidence that canonical
   actor order has reached Hero and reconciles any pending silent CHECKs;
6. still rejects two observable-impacting actions between frames as a true
   skipped transition;
7. rebuilds the canonical state from scratch after synchronization.

This matches the useful legacy DeepSpin lifecycle pattern: heartbeat does not
drive strategy computation; `DLLUpdateOnMyTurn` is the stable synchronization
point.

## Revised gate

The E2E audit now explicitly requires coverage of:

- deferred invisible CHECKs;
- later synchronization of those CHECKs at MyTurn or a changed frame;
- both domains;
- preflop and postflop Hero states;
- actual street reveals;
- corrupt frame rejection;
- true skipped-visible-transition rejection.

## Immediate action

```bash
bash tools/run_lt2_openholdem_observable_tracker_e2e.sh
```

Wait for `LT2_OPENHOLDEM_OBSERVABLE_TRACKER_E2E_PASS`, then send
`SpinCore_LT2_openholdem_observable_tracker_e2e.json`.

If it fails again, do not rerun: send the terminal output and JSON.
