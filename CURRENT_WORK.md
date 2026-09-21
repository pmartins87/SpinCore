# SpinCore Current Work

Date: 2026-09-21
Status: **HEARTBEAT/LIFECYCLE TRACKER PASS — OPENHOLDEM SYMBOL/SCRAPE ADAPTER NEXT**

## Frozen strategy/runtime

All strategic and model identities remain frozen.

No training, EV tuning or holdout reuse is permitted.

## Heartbeat/lifecycle result

PASS:
- 10,000 canonical transitions;
- 20,000 duplicate heartbeats;
- 7,862 MyTurn callbacks;
- 3,931 actual decision computations;
- 3,931 repeated MyTurn cache hits;
- 11,793 ProcessQuery reads;
- 1,170 NewRound callbacks;
- 0 action mismatches;
- 0 transcript mismatches.

Fail-closed faults:
- corrupt snapshots 500/500 rejected;
- wrong-hand snapshots 500/500 rejected;
- skipped transitions 493/493 rejected;
- failure latch 500/500;
- HandReset recovery 500/500.

## Active integration boundary

The state machine is now validated independently from OpenHoldem.

Next map actual OpenHoldem scrape/symbol semantics into the runtime's normalized table snapshot.

The adapter uses documented OH meanings:
- balanceN = stack behind;
- currentbetN = current betting-round chips in play;
- pot = total chips in play including player bets;
- playersdealtbits = hand participants;
- playersplayingbits = not-folded participants;
- playersallinbits = all-in participants;
- betround 1..4 = preflop..river;
- userchair/dealerchair and exact blinds;
- Hero + board cards converted to SpinCore ids.

Runtime canonicalization:
- logical 0 = dealer;
- 3H logical 1/2 = next two dealt chairs clockwise;
- HU logical 0 = dealer/SB, logical 1 = opponent/BB, logical 2 dead.

## Immediate action

```bash
bash tools/run_lt2_openholdem_symbol_adapter.sh
```

Wait for `LT2_OPENHOLDEM_SYMBOL_ADAPTER_PASS`, then send
`SpinCore_LT2_openholdem_symbol_adapter.json`.
