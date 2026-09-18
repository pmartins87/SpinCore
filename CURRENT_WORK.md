# SpinCore Current Work

Date: 2026-09-18
Status: **LT2 STAGE B PASS — TARGET-DRIFT MATRIX REVIEWED — JAMMER TARGET STATIONARY, PASSIVE FIT DEGRADATION NOT FAILURE-SPECIFIC, TURN MIXED — HU POLICY-CHAIN AUDIT NEXT — NO TRAINING**

## Active source of truth

Read before new compute:

- `docs/LT2_TARGET_DRIFT_TRACKING_RESULT_20260918.md`
- `docs/LT2_HU_POLICY_CHAIN_AUDIT_20260918.md`
- `docs/LT2_CROSS_STREET_FULL_JSON_REVIEW_20260918.md`
- `docs/LONG_TRAINING_PLAN.md`

Preserve Stage A and Stage B. Do not continue root training beyond iteration 7500.

## Preserved checkpoints

Stage A:
- iteration 3000 / 1.8M roots;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Target-drift / model-tracking result

### Jammer preflop after opponent all-in

Target drift is effectively zero:
- FAILURE `5.54e-17`;
- CONTROL `2.51e-16`;
- reference best action changed on 0/12 in both groups.

Stage-B own-target Advantage MSE did not worsen:
- FAILURE B-A `-0.001079`, CI `[-0.006408,+0.004250]`;
- CONTROL B-A `-0.001084`, CI `[-0.004754,+0.002585]`.

Therefore target nonstationarity cannot explain the deployed Jammer regression.

### PassiveCaller FLOP

Target drift is actually larger in CONTROL:
- FAILURE `0.000735`;
- CONTROL `0.001809`;
- F-C `-0.001074`, CI `[-0.001990,-0.000158]`.

Stage-B own-target model error worsens in selected FAILURE states:
- B-A `+0.000606`, CI `[+0.000189,+0.001023]`;
- 10/12 anchors worsen.

But the failure-control difference is unresolved:
- `+0.000131`, CI `[-0.001227,+0.001489]`.

Stage-B current Advantage best-action agreement with its own low-noise reference is 0/12 in both FAILURE and CONTROL.

### UniformLegal TURN

Target drift is large and heterogeneous:
- FAILURE mean `0.003490`;
- CONTROL mean `0.002354`;
- F-C unresolved.

Own-target B-A model error and tracking-error differences are also unresolved.

No single mechanism is established.

## Strategic interpretation

Do not return to K4 or target-estimator tuning.

The Jammer result moves the highest-value question downstream:

**Did Stage-B current Advantage behavior regress too, or did the loss appear mainly in historical AveragePolicy aggregation?**

This matters because the deployed benchmark uses AveragePolicy, while the target-drift audit inspected the current Advantage model.

## Active gate

Run:

```bash
bash tools/run_lt2_hu_policy_chain.sh
```

The audit uses the same forensic seed family only and compares, on identical HU scenarios/deals/RNG streams:

- Stage-A AveragePolicy;
- Stage-B AveragePolicy;
- Stage-A current Advantage-induced behavior;
- Stage-B current Advantage-induced behavior;

against:
- UNIFORM_LEGAL;
- PASSIVE_CALLER;
- JAMMER.

Primary contrasts:
- AVG B-A;
- BEH B-A;
- change in the AVG-vs-BEH gap.

Holdout `20261001..20261006` remains untouched.

DeepCrusher remains deferred.

## Immediate user action

Pull `main` and run:

```bash
bash tools/run_lt2_hu_policy_chain.sh
```

Wait for `LT2_HU_POLICY_CHAIN_EVAL_PASS` or the first error.

Then send `SpinCore_LT2_hu_policy_chain.json`.

Do not start any training.
