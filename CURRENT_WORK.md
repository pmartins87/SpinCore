# SpinCore Current Work

Date: 2026-09-21
Status: **FINAL HU HOLDOUT PASS — HYBRID PYTHON DEPLOYMENT PARITY EXACT — NATIVE C++ INFERENCE PARITY NEXT**

## Frozen strategic candidate

TRUE_HEADS_UP:
- current ENS8 @ iteration 8100.

THREE_HANDED:
- finalized AveragePolicy @ iteration 8100.

Strategic testing is closed.

## Frozen deployment artifact

`SpinCore_LT2_hybrid_deployment_8100.pt`

SHA256:

`87e46b40cb43bb89cb46bf3b760bbac5c8282491fd3d6da73d1bbe28329b278c`

Source:
- checkpoint `a51dbbed71090e45f2c4f5db6297f72eab848990b38650702b436e2aa60ca4bf`;
- HU ensemble `c44b817f75304db352eedc33febbd180e6d038e0580c91be0b9befdbdb08f181`.

## Deployment parity

PASS over:
- 3,000 hands;
- 10,599 decisions;
- 105,990 probability values;
- both 3H and HU.

Observed:
- max probability diff 0.0;
- mean probability diff 0.0;
- legal-context mismatches 0;
- argmax mismatches 0;
- exact action-resolution mismatches 0.

## Legacy-first runtime decision

Historical `deepspin/user_deepspin.cpp` was reviewed before OpenHoldem integration.

Preserve:
- infer on `DLLUpdateOnMyTurn`;
- cache decision for `ProcessQuery`;
- no strategic recompute on heartbeat;
- explicit hand/round lifecycle.

Do not preserve:
- duplicated manual 292-feature observation;
- silent zero-default state substitutions;
- independently reimplemented sizing semantics.

The LT2 bridge must use canonical solver state/observation/action semantics.

## Active gate

Native C++ inference parity.

This checks the cross-language model runtime needed by the eventual Windows/OpenHoldem user-DLL:
- SPNNIV1 decoding;
- embeddings;
- GRU;
- MLP;
- 3H masked softmax;
- HU ENS8 raw mean + lean regret matching.

No strategy change and no holdout reuse.

## Immediate action

```bash
bash tools/run_lt2_native_cpp_inference_parity.sh
```

Wait for `LT2_NATIVE_CPP_INFERENCE_PARITY_PASS`, then send
`SpinCore_LT2_cpp_inference_parity.json`.

Keep `SpinCore_LT2_cpp_deployment_8100.bin` in Downloads.
