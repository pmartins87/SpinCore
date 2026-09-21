# SpinCore — LT2 hybrid deployment parity result

Date: 2026-09-21  
Status: **PASS — PYTHON DEPLOYMENT BUNDLE FROZEN**

## Validated deployment bundle

SHA256:

`87e46b40cb43bb89cb46bf3b760bbac5c8282491fd3d6da73d1bbe28329b278c`

Source identities:

- iteration-8100 checkpoint:
  `a51dbbed71090e45f2c4f5db6297f72eab848990b38650702b436e2aa60ca4bf`;
- iteration-8100 HU ensemble:
  `c44b817f75304db352eedc33febbd180e6d038e0580c91be0b9befdbdb08f181`.

Bundle schema:

`SPINCORE_LT2_HYBRID_DEPLOYMENT_V1`

Deployment semantics:

- THREE_HANDED: finalized AveragePolicy;
- TRUE_HEADS_UP: 8-member raw-Advantage mean then unchanged lean regret matching.

## Mechanical parity evidence

The compact bundle was compared against the source artifacts on already-seen forensic trajectories.

Coverage:

- 3,000 hands;
- 1,658 THREE_HANDED hands;
- 1,342 TRUE_HEADS_UP hands;
- 10,599 decisions;
- 7,616 THREE_HANDED decisions;
- 2,983 TRUE_HEADS_UP decisions;
- 105,990 probabilities compared.

Result:

- legal-context mismatches: **0**;
- argmax mismatches: **0**;
- exact action-resolution mismatches: **0**;
- maximum absolute probability difference: **0.0**;
- mean absolute probability difference: **0.0**.

No strategic EV evaluation and no holdout reuse occurred.

## Decision

The PyTorch hybrid deployment artifact is mechanically identical to the validated source inference path on this gate and is now frozen.

Strategic testing remains closed.

The next step is a native C++ inference implementation intended for the OpenHoldem user-DLL path. This is a cross-language/runtime integration task only; it may not modify model weights, action semantics, domain routing, regret matching or policy selection.
