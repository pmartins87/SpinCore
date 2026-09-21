# SpinCore — Long-Training Plan

Status: **TRAINING CLOSED — NATIVE OPENHOLDEM PRODUCTIONIZATION**
Date: 2026-09-21

## Completed reference gates

- final strategic holdout PASS;
- Python deployment/source parity PASS;
- native C++ inference parity PASS;
- hidden-card filler invariance PASS;
- exact transcript rebuild PASS;
- public snapshot one-action reconciler PASS;
- heartbeat/lifecycle/cache tracker PASS;
- OpenHoldem symbol adapter PASS;
- full observable OpenHoldem E2E tracker PASS.

## Final reference evidence

The observable E2E gate completed:
- 10,000 transitions;
- 3,916 Hero canonical checks;
- 1,972 street reveals;
- 1,769 invisible CHECK deferrals;
- 719 MyTurn delayed reconciliations;
- 689 multi-action synchronization events;
- 500/500 corrupt-frame rejections;
- 500/500 skipped-transition rejections.

There were zero exact-action, transcript or canonical-state mismatches.

## Native productionization

The reference architecture is no longer the open question.

The next risk is implementation-language drift while porting that architecture
to the Windows C++ DLL.

A reusable native runtime module now owns:
- raw OpenHoldem-style frame normalization;
- logical seat mapping;
- blind-index validation;
- card-derived betround semantics;
- invisible CHECK handling;
- exact public transcript;
- hidden-card filler construction;
- from-scratch canonical rebuild;
- fail-closed synchronization.

The native audit must pass before model inference and OpenHoldem callbacks are
joined in the same DLL.

After native tracker PASS, no further strategic training is planned.
