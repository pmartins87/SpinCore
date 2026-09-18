# SpinCore — Long-Training Plan

Status: **LT2 STAGE B PASS — ROOT TRAINING PAUSED — TARGET-DRIFT MATRIX COMPLETE — HU POLICY-CHAIN AUDIT ACTIVE**
Date: 2026-09-18

## Preserved milestones

Stage A:
- iteration 3000 / 1.8M roots;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Never rewrite either checkpoint.

## Current scientific conclusion

Future-board K4 improves estimator quality but does not identify Stage-B failure states.

The Stage-A/B target-drift matrix adds:

### Jammer
- low-noise target is invariant A->B after opponent all-in;
- Stage-B own-target Advantage MSE does not worsen;
- deployed AveragePolicy nevertheless has a resolved negative B-A result.

Therefore the Jammer regression is not explained by target nonstationarity.

### PassiveCaller flop
- target drift is significantly larger in controls than failures;
- failures show an internal Stage-B own-target MSE increase;
- but its FAILURE-vs-CONTROL excess is unresolved;
- Stage-B Advantage best-action agreement is poor in both failure/control samples.

### UniformLegal turn
- target drift is substantial;
- own-target and tracking changes are heterogeneous;
- no failure-specific cause resolves.

## Highest-value distinction

The deployed benchmark evaluates AveragePolicy.

The current target/model diagnostics evaluate the current Advantage-induced behavior.

We now need to know whether Stage-B regression is:

1. already present in current behavior; or
2. introduced/amplified by historical AveragePolicy aggregation.

## HU policy-chain gate

Canonical contract:

`docs/LT2_HU_POLICY_CHAIN_AUDIT_20260918.md`.

Use forensic seeds `20260920..20260925`, 5000 scenarios/seed, HU only.

Evaluate:
- AVG_A;
- AVG_B;
- BEH_A;
- BEH_B;

against:
- UNIFORM_LEGAL;
- PASSIVE_CALLER;
- JAMMER.

Same scenario, deal, seat and random streams.

Measure:
- deployed AVG B-A;
- current BEH B-A;
- stage-specific AVG-BEH gaps;
- change in aggregation gap from A to B.

## Decision logic

If AVG B-A is negative while BEH B-A is neutral/positive:
- inspect policy reservoir/history weighting and AveragePolicy aggregation.

If both are negative:
- continue upstream through Advantage/self-play behavior.

If results split by baseline:
- accept a multi-mechanism diagnosis rather than forcing one global fix.

No training resumes before this gate is reviewed.

## Immediate direction

1. Keep Stage A/B frozen.
2. Run `bash tools/run_lt2_hu_policy_chain.sh`.
3. Wait for `LT2_HU_POLICY_CHAIN_EVAL_PASS`.
4. Send `SpinCore_LT2_hu_policy_chain.json`.
5. Keep holdout seeds `20261001..20261006` untouched.
6. Do not train K4 or resume long training.
