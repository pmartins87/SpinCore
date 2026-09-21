# SpinCore — LT2 native OpenHoldem tracker result

Date: 2026-09-21  
Status: **PASS — NATIVE C++ OBSERVABLE TRACKER VALIDATED**

## Scope

The C++ production tracker was exercised independently of Python.

No deployment-model inference, EV evaluation, optimizer work, training roots or holdout reuse occurred.

## Coverage

- public transitions: `12,000`;
- Hero canonical checks: `4,785`;
- invisible CHECK deferrals: `2,742`;
- delayed actions reconciled at MyTurn: `1,124`;
- multi-action synchronization events: `962`;
- street reveals: `3,660`.

## Fault rejection

- corrupt observable frames: `500/500` rejected;
- skipped observable transitions: `500/500` rejected.

Failures: **0**.

## Decision

The native C++ port preserves the accepted observable OpenHoldem reconstruction semantics:

- strict hand/chair/blind/card normalization;
- card-derived OpenHoldem betround;
- invisible CHECK deferral;
- MyTurn synchronization;
- canonical exact transcript;
- deterministic hidden fillers;
- from-scratch authoritative rebuild;
- fail-closed corruption/skip handling.

The next gate composes this native tracker with the already-validated frozen native neural runtime in **shadow mode only**.

No table action is enabled.
