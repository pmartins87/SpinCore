# SpinCore — Long-Training Plan

Status: **TRAINING CLOSED — NATIVE SHADOW DECISION INTEGRATION**
Date: 2026-09-21

## Completed runtime gates

- final strategic holdout PASS;
- Python deployment/source parity PASS;
- native C++ inference parity PASS;
- hidden-card filler invariance PASS;
- exact transcript rebuild PASS;
- public snapshot reconciler PASS;
- heartbeat/lifecycle/cache tracker PASS;
- OpenHoldem symbol adapter PASS;
- observable OpenHoldem E2E PASS;
- native C++ observable tracker PASS.

## Native tracker evidence

The C++ tracker completed:
- 12,000 transitions;
- 4,785 Hero checks;
- 2,742 invisible CHECK deferrals;
- 1,124 MyTurn delayed reconciliations;
- 962 multi-action synchronization events;
- 3,660 street reveals;
- 500/500 corrupt rejections;
- 500/500 skipped-transition rejections.

## Current composition gate

The new native shadow engine combines the validated tracker with the validated neural runtime while keeping action execution disabled.

It additionally enforces:
- exact frozen deployment metadata;
- external file SHA256 check before load;
- correct 3H/HU routing;
- exact seven-label lean active mask in the ten-slot carrier;
- zero probability on illegal actions;
- stable SplitMix64-based sampling rather than implementation-dependent `std::uniform_real_distribution`;
- exact lean-slot -> ExactAction resolution;
- repeated MyTurn cache identity;
- cache invalidation after canonical state mutation.

After PASS, proceed to the actual Windows OpenHoldem user-DLL ABI.
