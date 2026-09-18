# SpinCore Current Work

Date: 2026-09-18
Status: **LT2 STAGE B PASS — HU POLICY-CHAIN SPLIT RESOLVED — JAMMER DEFECT UPSTREAM IN CURRENT BEHAVIOR, PASSIVE AVG REGRESSION NOT REPRODUCED BY CURRENT BEHAVIOR — BEHAVIOR FIRST-DIVERGENCE NEXT — NO TRAINING**

## Active source of truth

Read before new compute:

- `docs/LT2_HU_POLICY_CHAIN_RESULT_20260918.md`
- `docs/LT2_HU_BEHAVIOR_FIRST_DIVERGENCE_20260918.md`
- `docs/LT2_TARGET_DRIFT_TRACKING_RESULT_20260918.md`
- `docs/LONG_TRAINING_PLAN.md`

Preserve Stage A and Stage B. Do not continue root training beyond iteration 7500.

## Preserved checkpoints

Stage A:
- iteration 3000 / 1.8M roots;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## HU policy-chain result

The global paired HU audit used 13,585 HU scenario clusters and 81,510 baseline/seat rows on forensic seeds only.

### JAMMER

AveragePolicy:
- A `-3.459`;
- B `-5.141`;
- B-A `-1.682`, CI95 `[-2.767,-0.597]`.

Current Advantage-induced behavior:
- A `+1.850`;
- B `-6.580`;
- B-A `-8.430`, CI95 `[-12.515,-4.345]`.

Aggregation-chain delta:
- `+6.748`, CI95 `[+2.521,+10.975]`.

Therefore the strong Jammer defect is already present upstream in current Advantage behavior. AveragePolicy buffers rather than amplifies the Stage-B-vs-A current-behavior loss.

This does not mean the single final Stage-B Advantage snapshot alone caused the historical AveragePolicy regression, because Advantage is reset/refit each iteration. It does mean the current Advantage/behavior chain contains a large resolved defect.

Combined with the prior stationary Jammer target result, the next target is action-ranking / regret-matching behavior, not K4 or target drift.

### PASSIVE_CALLER

AveragePolicy B-A:
- `-1.261`, CI95 `[-2.377,-0.145]`.

Current behavior B-A:
- `+2.083`, CI95 `[-1.413,+5.579]`.

Aggregation-chain delta:
- `-3.344`, CI95 `[-6.944,+0.255]`.

The deployed regression is not reproduced by the final current behavior. Historical aggregation / policy reservoir remains a separate candidate, but the chain delta is unresolved.

### UNIFORM_LEGAL

AveragePolicy B-A:
- `-0.416`, unresolved.

Current behavior B-A:
- `-3.461`, unresolved.

Aggregation-chain delta:
- `+3.045`, unresolved.

## Strategic interpretation

There is no single Stage-B mechanism.

Highest priority is the resolved Jammer current-behavior loss because:
- it is large;
- it is upstream of AveragePolicy;
- its previously audited facing-all-in target is stationary A->B;
- own-target MSE did not show a matching global degradation.

The next question is therefore:

**Where does the -8.43 chips/hand current-behavior Jammer loss first manifest?**

Do not train a fix before localizing it.

## Active gate

Run:

```bash
bash tools/run_lt2_hu_behavior_first_divergence.sh
```

It replays Stage A/B current Advantage-induced behavior in lock-step under the same forensic HU scenarios, deals, baselines, hero seats and RNG streams.

First-divergence groups:
- NO_DIVERGENCE;
- PREFLOP_ROOT;
- PREFLOP_FACING_ALL_IN;
- PREFLOP_OTHER;
- FLOP;
- TURN;
- RIVER.

If Jammer loss again concentrates in PREFLOP_FACING_ALL_IN, the next audit will test broad non-selected action-gap / regret-matching calibration there.

Holdout `20261001..20261006` remains untouched.

DeepCrusher remains deferred.

## Immediate user action

Pull `main` and run:

```bash
bash tools/run_lt2_hu_behavior_first_divergence.sh
```

Wait for `LT2_HU_BEHAVIOR_FIRST_DIVERGENCE_PASS` or the first error.

Then send `SpinCore_LT2_hu_behavior_first_divergence.json`.

Do not start any training.
