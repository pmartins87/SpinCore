# SpinCore Current Work

Date: 2026-09-20
Status: **ITERATION 8000 FROZEN — AVERAGEPOLICY JAMMER IMPROVEMENT RESOLVED — CURRENT BEHAVIOR PASSIVE REGRESSION RESOLVED — READ-ONLY LOCALIZATION NEXT**

## Preserved checkpoints

Stage B:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

HU400 pilot:
- iteration 7600 / 4.56M roots;
- SHA256 `c34f19802d3ad5ad3a5d131b083ffcee0e0867667ae0d57324cf35d5fa246b80`.

HU400 refresh:
- iteration 8000 / 4.80M roots;
- SHA256 `773b5d523c7fc5fcbfc3d10cb1f5be6429e50f4283259df9134963db8d274886`.

## Deployment AveragePolicy — positive result

Stage B -> 8000:

- JAMMER: `+1.9021`, CI95 `[+0.6874,+3.1168]` — resolved improvement;
- PASSIVE_CALLER: `+0.7865`, unresolved — no regression;
- UNIFORM_LEGAL: `+1.1757`, unresolved — no regression.

The deployment policy has begun to absorb the HU400 repair.

## Current Advantage behavior — new tradeoff

Stage B -> 8000:

- JAMMER: `+5.5310`, resolved improvement;
- PASSIVE_CALLER: `-2.9214`, CI95 `[-5.5408,-0.3020]` — resolved regression;
- UNIFORM_LEGAL: `-0.9510`, unresolved.

The preregistered current-behavior no-tradeoff condition therefore fails.

## Decision

Do not continue roots.

Freeze iteration 8000 as a deployment candidate, but do not unseal holdout yet.

Next run is read-only: compare iteration 7600 directly with 8000 and localize the first current-behavior divergence, with PASSIVE_CALLER as the primary target.

## Immediate user action

Pull main and run:

```bash
bash tools/run_lt2_hu_7600_8000_behavior_first_divergence.sh
```

Wait for `LT2_HU_7600_8000_BEHAVIOR_FIRST_DIVERGENCE_PASS`.

Then send `SpinCore_LT2_hu_7600_8000_behavior_first_divergence.json`.

Do not train beyond iteration 8000.
