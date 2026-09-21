# SpinCore — Long-Training Plan

Status: **TRAINING CLOSED — OPENHOLDEM SCRAPE NORMALIZATION**
Date: 2026-09-21

## Completed runtime gates

- final strategic holdout PASS;
- Python deployment/source parity PASS;
- native C++ inference parity PASS;
- hidden-card filler invariance PASS;
- exact public-transcript rebuild PASS;
- one-action public snapshot reconciler PASS;
- heartbeat/lifecycle/cache tracker PASS.

## Heartbeat evidence

The lifecycle gate covered:
- 10,000 valid transitions;
- 20,000 duplicate heartbeats;
- 3,931 distinct Hero computations;
- 11,793 ProcessQuery cache reads;
- 1,170 round transitions.

All action/transcript/cache checks passed.

Fault rejection was exact for every generated corrupt, wrong-hand and skipped
transition.

## OpenHoldem scrape adapter

The next layer is intentionally strict.

At HandReset/preflop anchor:
- require exactly two or three dealt chairs;
- require user and dealer among them;
- canonicalize physical chairs to LT2 logical seats;
- recover exact starting stacks as balance + currentbet;
- verify the observed blind posts against the authoritative betting rules;
- map exact blind pair to the frozen legacy blind index.

During hand:
- chair/dealer/hand/blinds/dealt set must remain stable;
- balances/currentbets/pot must be integral chip values;
- board count must match betround;
- Hero cards cannot change;
- all-in/playing/dealt bitsets must be internally consistent;
- pot must equal hand-start chips minus current remaining balances.

After this adapter passes, replace the reconciler's synthetic full
PublicSnapshot input with the actually observable OpenHoldem projection.
