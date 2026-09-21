# SpinCore Roadmap — active state 2026-09-21

## Active status

- strategic candidate ENS8@8100 — **FROZEN / HOLDOUT PASS**;
- Python deployment parity — **PASS EXACT**;
- native C++ inference parity — **PASS**;
- OpenHoldem legacy-first review — **COMPLETE**;
- hidden-card/future-board filler invariance — **NEXT**;
- live OpenHoldem shadow-state reconstruction — **AFTER FILLER GATE**;
- user-DLL action bridge — **AFTER STATE RECONSTRUCTION**.

## Native runtime result

Standalone C++ reproduces the frozen deployment model with zero argmax mismatch and max probability drift only `2.527e-05`.

## Current blocker

The next issue is reconstructing the exact canonical public solver state from observable table information.

The proposed bridge uses deterministic fillers only for information that must be strategically invisible at the current decision.

That invariance is now tested explicitly before building the live tracker.
