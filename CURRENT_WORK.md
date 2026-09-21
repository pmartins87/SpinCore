# SpinCore Current Work

Date: 2026-09-21
Status: **EXACT TRANSCRIPT REBUILD PASS — PUBLIC SNAPSHOT ACTION RECONCILER NEXT**

## Frozen strategy/runtime

Strategic candidate and model identities remain frozen.

No training, EV tuning or holdout reuse is permitted.

## Canonical rebuild result

PASS:
- 5,000 Hero decision states;
- 20,000 alternate from-scratch rebuilds;
- 83,724 legal exact-action comparisons;
- both 3H and HU;
- all four streets;
- public transcript length up to 15 voluntary actions.

Rebuild inputs:
- original scenario;
- Hero cards;
- visible board;
- hidden-card fillers;
- exact public voluntary transcript.

Replaying that transcript through the authoritative exact-action solver API reproduced the canonical state exactly.

## Remaining OpenHoldem problem

The runtime no longer needs a persistent hidden deal.

It needs a reliable event tracker that converts successive public snapshots into the exact canonical transcript.

The validated LT2 convention must normalize economically equivalent aliases:
- all-in call -> CALL;
- stack-emptying aggression -> ALL_IN;
- non-all-in opening aggression -> BET_TO;
- non-all-in aggression facing a bet -> RAISE_TO.

Skipped or corrupted snapshots must fail closed rather than inventing a transcript.

## Active gate

A public runtime snapshot ABI and deterministic one-action reconciler are now implemented.

The mechanical gate generates diverse exact actions, arbitrary legal raise sizes, all-in aliases and fault injections.

## Immediate action

```bash
bash tools/run_lt2_public_snapshot_reconciler.sh
```

Wait for `LT2_PUBLIC_SNAPSHOT_RECONCILER_PASS`, then send
`SpinCore_LT2_public_snapshot_reconciler.json`.
