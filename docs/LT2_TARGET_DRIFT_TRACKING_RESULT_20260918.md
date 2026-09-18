# SpinCore — LT2 Stage-A/B target-drift + model-tracking result

Date: 2026-09-18
Status: **PASS — TARGET NONSTATIONARITY DOES NOT EXPLAIN JAMMER; PASSIVE FLOP SHOWS STAGE-B OWN-TARGET FIT DEGRADATION BUT NOT FAILURE-SPECIFIC; TURN REMAINS MIXED — POLICY-CHAIN EVALUATION NEXT**

## Integrity

The report `SPINCORE_LT2_CROSS_STREET_TARGET_DRIFT_TRACKING_V1` contains:

- 72 anchors;
- 12 FAILURE + 12 CONTROL per context;
- same forensic seeds `20260920..20260925`;
- same hidden deal, future boards and target RNG paired A/B;
- 8 future boards × 4 repeats;
- exact opponent level 1;
- canonical action-gap gauge;
- no training roots;
- no optimizer steps;
- no memory writes;
- holdout `20261001..20261006` untouched.

## JAMMER preflop after opponent all-in

The conditional target is effectively invariant from Stage A to Stage B:

- FAILURE target-drift MSE `5.54e-17`;
- CONTROL target-drift MSE `2.51e-16`;
- reference best action changed on `0/12` FAILURE and `0/12` CONTROL anchors.

Therefore **target nonstationarity cannot explain the Jammer regression at these states**.

Own-target Advantage-model error:

FAILURE:
- Stage A `0.013684`;
- Stage B `0.012605`;
- B-A `-0.001079`, CI `[-0.006408,+0.004250]`.

CONTROL:
- Stage A `0.008076`;
- Stage B `0.006991`;
- B-A `-0.001084`, CI `[-0.004754,+0.002585]`.

No Stage-B own-target MSE degradation is detected.

Yet the deployed AveragePolicy has a resolved negative Jammer B-A benchmark result.

This strongly moves the Jammer investigation downstream from **target generation** toward the **current-behavior -> AveragePolicy aggregation/deployment chain**.

Important caution: MSE can improve while regret-matching action ranking changes adversely. A direct policy-space evaluation is still required.

## PASSIVE_CALLER flop

Target drift:

FAILURE:
- `0.000735`, CI `[0.000328,0.001142]`.

CONTROL:
- `0.001809`, CI `[0.000989,0.002630]`.

FAILURE minus CONTROL:
- `-0.001074`;
- CI `[-0.001990,-0.000158]`.

Thus target drift is actually **larger in CONTROL states**. Target nonstationarity is not a failure discriminator here.

Stage-B minus Stage-A own-target model error:

FAILURE:
- `+0.000606`;
- CI `[+0.000189,+0.001023]`;
- 10 of 12 anchors worsened.

CONTROL:
- `+0.000475`;
- CI `[-0.000818,+0.001767]`.

FAILURE minus CONTROL:
- `+0.000131`;
- CI `[-0.001227,+0.001489]`.

So there is a real Stage-B fit degradation inside the selected flop FAILURE states, but it is **not resolved as failure-specific relative to controls**.

Reference best action changed A->B on:
- 7/12 FAILURE;
- 8/12 CONTROL.

Current Advantage argmax matched its own low-noise reference on:
- Stage A: 2/12 FAILURE and 2/12 CONTROL;
- Stage B: 0/12 FAILURE and 0/12 CONTROL.

This reveals a broader action-ranking weakness in these difficult flop states, not a clean failure-only mechanism.

## UNIFORM_LEGAL turn

Target drift is substantial and heterogeneous:

FAILURE:
- mean `0.003490`;
- median `0.000561`;
- reference best action changed in 9/12.

CONTROL:
- mean `0.002354`;
- median `0.000824`;
- reference best action changed in 6/12.

FAILURE minus CONTROL target-drift difference:
- `+0.001136`;
- CI `[-0.002786,+0.005059]`.

Unresolved.

Own-target model error B-A:

FAILURE:
- `+0.000210`;
- CI `[-0.002017,+0.002437]`.

CONTROL:
- `-0.001416`;
- CI `[-0.003423,+0.000591]`.

FAILURE minus CONTROL:
- `+0.001626`;
- CI `[-0.001372,+0.004624]`.

Unresolved.

Tracking-error difference FAILURE-CONTROL is also unresolved.

Therefore the turn regression cannot yet be assigned to target drift, approximation failure, or both.

## Overall verdict

The matrix rejects a single universal mechanism.

### Jammer
- target is stationary;
- Stage-B own-target MSE does not worsen;
- deployed AveragePolicy nevertheless regresses.

This points to the **policy aggregation/deployment chain**.

### Passive flop
- target drift is larger in controls;
- selected failures do show a resolved Stage-B own-target MSE increase;
- but the increase is not failure-specific versus controls;
- both failure and control samples have poor current Advantage best-action agreement.

This suggests a broader function-approximation / action-ranking issue may exist, but it does not yet explain why only particular deployed paths lose EV.

### Uniform turn
- strong target movement exists;
- model-tracking evidence is mixed;
- no failure-specific mechanism is resolved.

## Highest-value next test

Do not keep drilling target-estimator noise.

Run a **global HU policy-chain evaluation** on the exact forensic seed family.

Evaluate four hero policies on the same scenarios/deals/RNG streams against:
- UNIFORM_LEGAL;
- PASSIVE_CALLER;
- JAMMER.

Policies:
1. Stage-A deployed AveragePolicy;
2. Stage-B deployed AveragePolicy;
3. Stage-A current Advantage-induced behavior policy;
4. Stage-B current Advantage-induced behavior policy.

This directly asks:

**Did the current Stage-B behavior itself regress, or did the regression appear only after historical AveragePolicy aggregation?**

Decision:

- AveragePolicy B-A negative but Behavior B-A neutral/positive -> aggregation/history path is implicated;
- both AveragePolicy and Behavior B-A negative -> upstream Advantage/self-play mechanism;
- baseline-dependent split -> multiple mechanisms.

Holdout stays sealed. No training is authorized.
