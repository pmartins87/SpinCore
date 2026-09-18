# SpinCore — LT2 HU policy-chain result

Date: 2026-09-18
Status: **PASS — MULTI-MECHANISM; JAMMER REGRESSION IS ALREADY PRESENT IN CURRENT ADVANTAGE BEHAVIOR; PASSIVE AVG REGRESSION IS NOT PRESENT IN CURRENT BEHAVIOR; UNIFORM UNRESOLVED**

## Integrity

Report schema: `SPINCORE_LT2_HU_POLICY_CHAIN_EVAL_V1`.

The audit used:

- forensic seeds `20260920..20260925`;
- 5000 scenarios/seed;
- 13,585 HU scenario clusters;
- 81,510 baseline/seat rows;
- identical HU scenario, deal, hero seat and RNG streams across AVG_A, AVG_B, BEH_A and BEH_B;
- no new training roots;
- no optimizer steps;
- no training-memory writes;
- holdout `20261001..20261006` untouched.

## JAMMER

Deployed AveragePolicy:

- Stage A: `-3.459` chips/hand, CI95 `[-6.398,-0.520]`;
- Stage B: `-5.141`, CI95 `[-8.066,-2.216]`;
- B-A: `-1.682`, CI95 `[-2.767,-0.597]`.

Current Advantage-induced behavior:

- Stage A: `+1.850`, CI95 `[-1.054,+4.754]`;
- Stage B: `-6.580`, CI95 `[-9.935,-3.225]`;
- B-A: `-8.430`, CI95 `[-12.515,-4.345]`.

Aggregation-chain delta:

`(AVG_B-BEH_B) - (AVG_A-BEH_A) = +6.748`,
CI95 `[+2.521,+10.975]`.

Interpretation:

The Jammer deterioration is **already present upstream in the current Advantage-induced behavior**, and it is much larger there than in deployed AveragePolicy.

AveragePolicy does not amplify the Jammer failure. On this metric it actually **buffers** approximately 6.75 chips/hand of the Stage-B-vs-A current-behavior deterioration.

This does not prove that the final Stage-B Advantage snapshot alone caused the historical AveragePolicy loss, because the Advantage network is reset/refit each iteration. It does prove that the present Stage-B Advantage/behavior chain contains a strong Jammer defect.

Combined with the prior result that Jammer facing-all-in low-noise targets are A/B stationary and own-target MSE does not globally worsen, the next question is about **where and how action ranking / regret-matching behavior degrades**, not about target drift or AveragePolicy aggregation.

## PASSIVE_CALLER

Deployed AveragePolicy:

- Stage A: `+0.265`, CI95 `[-2.282,+2.812]`;
- Stage B: `-0.996`, CI95 `[-3.551,+1.558]`;
- B-A: `-1.261`, CI95 `[-2.377,-0.145]`.

Current behavior:

- Stage A: `+0.632`, CI95 `[-2.155,+3.419]`;
- Stage B: `+2.716`, CI95 `[+0.368,+5.063]`;
- B-A: `+2.083`, CI95 `[-1.413,+5.579]`.

Aggregation-chain delta:

- `-3.344`, CI95 `[-6.944,+0.255]`.

Interpretation:

The resolved deployed AveragePolicy regression is **not reproduced by the final current behavior**, which is directionally better at Stage B.

The aggregation-chain delta points negative but narrowly crosses zero.

Therefore PassiveCaller is a separate mechanism candidate involving historical behavior aggregation / policy reservoir / deployment, but it is not yet causally resolved.

## UNIFORM_LEGAL

Deployed AveragePolicy B-A:

- `-0.416`, CI95 `[-1.812,+0.979]`.

Current behavior B-A:

- `-3.461`, CI95 `[-7.496,+0.574]`.

Aggregation-chain delta:

- `+3.045`, CI95 `[-1.218,+7.307]`.

All primary A/B contrasts are unresolved.

## Overall verdict

A single Stage-B failure mechanism is rejected.

### Jammer
Strong current-Advantage/behavior regression exists.

### PassiveCaller
Deployed AveragePolicy regresses, but final current behavior does not reproduce that regression.

### UniformLegal
No resolved A/B mechanism.

## Next gate

Localize the **current-behavior** Stage-A/B first divergence, using the same exact paired decomposition previously used for AveragePolicy.

Primary question:

**Where does the `-8.43` chips/hand Jammer BEH_B-BEH_A loss first appear?**

Groups:

- NO_DIVERGENCE;
- PREFLOP_ROOT;
- PREFLOP_FACING_ALL_IN;
- PREFLOP_OTHER;
- FLOP;
- TURN;
- RIVER.

If the current-behavior Jammer loss is again concentrated in `PREFLOP_FACING_ALL_IN`, the next audit will target broad, non-selected action-gap / regret-matching calibration there.

If it localizes elsewhere, follow that state class instead.

No training is authorized.
