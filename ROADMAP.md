# SpinCore Roadmap — active state 2026-09-21

## Active status

- strategic training for validated HU candidate — **CLOSED**;
- final sealed HU holdout — **PASS 11/11**;
- frozen hybrid deployment bundle — **CREATED**;
- Python deployment/source parity — **PASS EXACT**;
- legacy OpenHoldem runtime review — **COMPLETE**;
- native C++ inference parity — **NEXT**;
- OpenHoldem state/lifecycle integration — **AFTER NATIVE PARITY**.

## Frozen identity

Hybrid deployment bundle SHA256:

`87e46b40cb43bb89cb46bf3b760bbac5c8282491fd3d6da73d1bbe28329b278c`

No model or strategy change is allowed.

## Native runtime phase

The legacy embedded-inference architecture is retained in principle, but the old duplicated observation feature logic is not.

First prove that a standalone C++ implementation reproduces the frozen Python deployment model on canonical SPNNIV1 fixtures.

Then reuse that exact C++ inference core inside the OpenHoldem user-DLL bridge.

## After native parity

Implement:
- OpenHoldem lifecycle callbacks;
- canonical state acquisition/reconstruction;
- exact HU/3H domain routing;
- canonical lean legal/action resolver;
- fail-closed invalid-state barrier;
- reproducible mixed-strategy sampling and decision audit;
- cached `ProcessQuery` outputs.

No table authorization follows from native parity alone.
