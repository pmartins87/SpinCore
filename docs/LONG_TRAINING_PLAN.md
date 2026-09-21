# SpinCore — Long-Training Plan

Status: **TRAINING CLOSED — OPENHOLDEM LIFECYCLE/CACHE INTEGRATION**
Date: 2026-09-21

## Deployment progress

Completed:
- final strategic holdout;
- Python deployment/source parity;
- native C++ inference parity;
- hidden-card filler invariance;
- exact transcript rebuild;
- public snapshot one-action reconciliation.

## Reconciler evidence

12,000 transitions covered all six exact action types and both domains.

Canonical aliases were validated with zero failures:
- all-in call -> CALL;
- stack-emptying aggression -> ALL_IN.

## Lifecycle layer

The next tracker is strategy-agnostic.

Its sole job is to guarantee reliable state/event semantics around OpenHoldem callbacks:

1. duplicate heartbeat does not mutate transcript;
2. exactly one valid transition appends exactly one action;
3. NewRound never resets hand transcript;
4. MyTurn computes once for the current canonical state;
5. repeated ProcessQuery is cache-only;
6. any state change invalidates the cache;
7. invalid hand identity, corrupted transition or skipped transition fails closed;
8. a failure remains latched until HandReset;
9. HandReset permits clean recovery.

This gate uses only old forensic data and no model inference.

After PASS, implement the actual OpenHoldem scraper/symbol normalization layer.
