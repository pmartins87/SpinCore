# SpinCore — LT2 HU current-behavior first-divergence forensic

Date: 2026-09-18
Status: **ACTIVE — LOCALIZE THE RESOLVED STAGE-B CURRENT-BEHAVIOR JAMMER LOSS**

## Trigger

The global HU policy-chain audit resolved:

- JAMMER AveragePolicy B-A: `-1.682`, CI95 `[-2.767,-0.597]`;
- JAMMER current behavior B-A: `-8.430`, CI95 `[-12.515,-4.345]`;
- aggregation-chain delta: `+6.748`, CI95 `[+2.521,+10.975]`.

Therefore the Jammer defect is already present in the Stage-B current Advantage-induced behavior, and AveragePolicy buffers rather than amplifies it.

The previous target-drift matrix also showed that Jammer facing-all-in conditional targets are stationary A->B and own-target MSE does not globally worsen.

The remaining high-value question is:

**Where does the current-behavior loss first manifest?**

## Design

Same forensic seed family:

`20260920..20260925`.

Same 5000 scenarios/seed.

HU only.

For every baseline and hero seat:
- identical scenario;
- identical deal;
- identical opponent policy;
- identical RNG streams;
- Stage A and Stage B current Advantage-induced behavior run in lock-step until the first sampled hero action differs.

Classify exactly one first-divergence group:

- `NO_DIVERGENCE`;
- `PREFLOP_ROOT`;
- `PREFLOP_FACING_ALL_IN`;
- `PREFLOP_OTHER`;
- `FLOP`;
- `TURN`;
- `RIVER`.

The group chip-EV contributions add back exactly to total paired B-A.

## Decision

If Jammer current-behavior loss is concentrated again in `PREFLOP_FACING_ALL_IN`:

- next audit broad, non-selected action-gap / regret-matching calibration there;
- measure raw Advantage ranking, positive-support regime, model/reference action gaps and policy regret.

If it is concentrated elsewhere:
- follow the observed state class instead.

This avoids tuning a fix to the previously selected AveragePolicy divergences.

## Launcher

```bash
bash tools/run_lt2_hu_behavior_first_divergence.sh
```

Expected marker:

`LT2_HU_BEHAVIOR_FIRST_DIVERGENCE_PASS`.

Holdout remains sealed.

No training is authorized.
