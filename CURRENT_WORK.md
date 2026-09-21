# SpinCore Current Work

Date: 2026-09-21
Status: **OPENHOLDEM SYMBOL ADAPTER PASS — OBSERVABLE END-TO-END TRACKER NEXT**

## Frozen strategy/runtime

All strategic/model identities remain frozen.

No training, EV tuning or holdout reuse is permitted.

## OpenHoldem symbol adapter result

PASS:
- 1,271 hand anchors;
- 9,000 runtime frames;
- 370 distinct physical 0..9 chair layouts;
- both 3H and HU;
- all four streets;
- all three Hero logical positions.

Fault rejection:
- wrong hand 300/300;
- unsupported blind level 300/300;
- fractional chip value 300/300;
- bad board-count/betround combination 300/300.

## Validated raw semantics

The adapter now has a strict contract for:
- balanceN;
- currentbetN;
- pot;
- dealt/playing/all-in bitsets;
- userchair/dealerchair;
- blinds;
- betround;
- Hero/board cards.

Physical chairs are canonicalized to LT2 logical seats:
- logical 0 = dealer/button;
- 3H logical 1/2 = next dealt chairs clockwise;
- HU logical 0 = dealer/SB, logical 1 = opponent/BB, logical 2 dead.

## Active gate

The previous reconciler used a full solver PublicSnapshot, which contains more
information than OpenHoldem truly exposes.

The new observable tracker consumes only the strict OpenHoldem projection.

On every accepted public action it:
1. infers one canonical exact action from observable stack/bet/fold changes;
2. appends that action to the transcript;
3. discards the old filler deal;
4. rebuilds the authoritative state from scratch with actual Hero cards and
   currently visible board;
5. fills only still-hidden cards deterministically;
6. replays the exact transcript;
7. demands canonical parity at Hero decisions.

This specifically validates flop/turn/river reveals replacing earlier filler
cards.

## Immediate action

```bash
bash tools/run_lt2_openholdem_observable_tracker_e2e.sh
```

Wait for `LT2_OPENHOLDEM_OBSERVABLE_TRACKER_E2E_PASS`, then send
`SpinCore_LT2_openholdem_observable_tracker_e2e.json`.
