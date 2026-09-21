# SpinCore — Long-Training Plan

Status: **TRAINING CLOSED — OPENHOLDEM PUBLIC EVENT TRACKING**
Date: 2026-09-21

## Deployment progress

Completed:
- strategic holdout PASS;
- Python deployment/source parity PASS;
- native C++ inference parity PASS;
- hidden-card filler invariance PASS;
- exact public-transcript rebuild PASS.

## Proven rebuild property

The canonical state can be reconstructed from scratch at Hero decisions without retaining hidden opponent/future cards across streets.

This removes the stale-filler-board problem.

## Current integration gate

The new runtime public snapshot exposes:
- terminal/street/actor/domain;
- pot/current bet;
- per-seat stacks;
- per-seat street/total commitments;
- folded/all-in flags;
- current legal exact-action bounds.

The reconciler derives the canonical exact action from one public transition and verifies it by applying the action through the authoritative solver and demanding an exact public-snapshot match.

Canonicalization:
- all-in call -> CALL;
- stack-emptying aggression -> ALL_IN;
- otherwise BET_TO/RAISE_TO according to the pre-action state.

Fault injection must prove that no-op, corrupted and skipped-action snapshots fail closed.

After PASS, implement the OpenHoldem heartbeat shadow tracker and cached Hero-decision interface.
