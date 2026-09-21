# SpinCore — Long-Training Plan

Status: **TRAINING CLOSED — DEPLOYMENT ENGINEERING ONLY**
Date: 2026-09-21

## Strategic closure

The validated candidate passed:
- post-training weak baselines;
- independent learned-policy crossplay;
- final sealed holdout;
- exact Python deployment/source parity.

The strategic candidate is frozen.

## Deployment artifact

SHA256:

`87e46b40cb43bb89cb46bf3b760bbac5c8282491fd3d6da73d1bbe28329b278c`

Semantics:
- 3H AveragePolicy;
- HU current ENS8@8100.

## Native C++ step

The next runtime implementation follows the useful legacy DeepSpin pattern of local inference inside the OpenHoldem bridge.

Before any OpenHoldem symbol/state integration, the neural runtime itself must match Python.

Native implementation covers:
- SPNNIV1 decode;
- embeddings;
- PyTorch-compatible GRU math;
- dense layers;
- 3H masked softmax;
- HU eight-member raw Advantage average;
- lean regret matching.

Parity fixtures use old forensic seeds only and contain no holdout evaluation.

PASS requires:
- both domains covered;
- preflop and postflop covered;
- zero argmax mismatches;
- zero illegal mass;
- finite outputs;
- max probability drift within the committed numerical tolerance.

After PASS, continue to actual OpenHoldem bridge construction. Do not run more strategic tests.
