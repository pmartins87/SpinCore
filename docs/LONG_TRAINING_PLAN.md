# SpinCore — Long-Training Plan

Status: **TRAINING CLOSED — OPENHOLDEM OBSERVABLE END-TO-END INTEGRATION**
Date: 2026-09-21

## Completed runtime gates

- final strategic holdout PASS;
- Python deployment/source parity PASS;
- native C++ inference parity PASS;
- hidden filler invariance PASS;
- exact transcript rebuild PASS;
- public snapshot one-action reconciler PASS;
- heartbeat/lifecycle/cache tracker PASS;
- OpenHoldem symbol/scrape adapter PASS.

## Symbol adapter evidence

The strict OpenHoldem normalization gate passed 9,000 frames across 370 physical
chair layouts, both domains and all streets with zero mismatches.

All 1,200 malformed frames were rejected.

## Final pre-DLL integration gate

Real OpenHoldem does not expose the solver's full PublicSnapshot.

The observable tracker therefore works only from:
- hand anchor;
- normalized balances/current bets/pot;
- dealt/playing/all-in status;
- visible board;
- Hero cards.

For each changed frame:
1. infer the canonical exact action;
2. append to public transcript;
3. rebuild state from hand start with actual visible cards;
4. replace hidden cards with deterministic fillers;
5. replay transcript through authoritative exact-action API.

At Hero turns, the result must exactly match authoritative SPNNIV1/SPNNIV2 and
lean action semantics.

Once this passes, the architecture is ready to be translated to the actual
Windows OpenHoldem user-DLL callbacks.
