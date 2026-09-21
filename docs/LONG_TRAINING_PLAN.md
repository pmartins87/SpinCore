# SpinCore — Long-Training Plan

Status: **TRAINING CLOSED — OPENHOLDEM CANONICAL STATE REBUILD**
Date: 2026-09-21

## Deployment progress

Completed:
- strategic holdout PASS;
- Python deployment/source parity PASS;
- native C++ inference parity PASS;
- hidden-card filler invariance PASS.

## Proven filler property

Across 6,000 states and 36,000 alternate completions, opponent private cards and unrevealed future board cards did not affect the current canonical inference/action state.

This permits deterministic legal fillers for information unavailable to OpenHoldem.

## Rebuild rather than mutate hidden board

A filler chosen preflop may not equal the real turn/river card later revealed.

The runtime should therefore not rely on one persistent hidden complete deal across the whole hand.

Instead, at each Hero decision:
1. instantiate the original scenario again;
2. set Hero cards and every currently visible board card exactly;
3. fill only still-hidden card positions deterministically;
4. replay the accumulated exact public action transcript;
5. ask the authoritative solver for canonical observation/legal/action resolution;
6. run the frozen native inference model.

## Current mechanical gate

Exact transcript replay must reproduce:
- actor/domain;
- SPNNIV1;
- SPNNIV2;
- legal lean action set;
- exact resolver output for every current legal action.

Only after this passes should the OpenHoldem snapshot/history reconciliation layer be implemented.
