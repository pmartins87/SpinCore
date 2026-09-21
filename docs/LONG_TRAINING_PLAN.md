# SpinCore — Long-Training Plan

Status: **TRAINING CLOSED — OPENHOLDEM RUNTIME STATE INTEGRATION**
Date: 2026-09-21

## Completed deployment gates

- final strategic holdout: PASS;
- Python compact bundle/source parity: exact PASS;
- native C++ model inference parity: PASS.

No further strategic training/testing is planned for this candidate.

## Current runtime question

The authoritative solver requires a complete card deal to instantiate a state.

OpenHoldem only exposes:
- Hero private cards;
- visible public board;
- public betting/table state.

The bridge therefore needs filler values for strategically hidden card positions.

This is safe only if those fillers cannot alter the current canonical observation, legal actions or exact action resolution.

## Mechanical prerequisite

Use old forensic trajectories and repeatedly replace:
- opponent private holes;
- unrevealed future board.

Preserve:
- current actor's Hero cards;
- visible board;
- scenario;
- complete public lean action path.

Require exact equality of:
- actor/domain;
- SPNNIV1;
- SPNNIV2 public metadata;
- legal action set;
- every legal action's exact resolver output.

After PASS, implement the live OpenHoldem shadow-state tracker with fail-closed ambiguity handling.
