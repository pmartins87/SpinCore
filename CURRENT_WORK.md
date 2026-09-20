# SpinCore Current Work

Date: 2026-09-20
Status: **ITERATION 8000 FROZEN — PASSIVE/UNIFORM CURRENT-BEHAVIOR LOSS LOCALIZED TO PREFLOP ROOT — DETERMINISTIC ROOT-POLICY AUDIT NEXT**

## Preserved checkpoints

Stage B 7500:
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

HU400 pilot 7600:
- SHA256 `c34f19802d3ad5ad3a5d131b083ffcee0e0867667ae0d57324cf35d5fa246b80`.

HU400 refresh 8000:
- SHA256 `773b5d523c7fc5fcbfc3d10cb1f5be6429e50f4283259df9134963db8d274886`.

## 7600 -> 8000 first-divergence result

PASSIVE_CALLER:
- total `-4.2843`, resolved;
- PREFLOP_ROOT contribution `-3.0561`, CI95 `[-5.2509,-0.8613]` — dominant resolved component.

UNIFORM_LEGAL:
- total `-9.3743`, resolved;
- PREFLOP_ROOT contribution `-7.4686`, CI95 `[-10.5478,-4.3895]` — dominant resolved component.

JAMMER:
- total `+1.3480`, unresolved;
- FAI `+2.2141`, root `-0.8661`, both unresolved.

The original FAI repair survives directionally. The new failure is preflop-root drift.

## Root transition clue

Most common root first-divergence:

- `POT_33 -> ALL_IN`: 5,618 seat-runs.

Also:
- `CHECK_CALL -> ALL_IN`: 1,546;
- `FOLD -> ALL_IN`: 914.

This strongly suggests a new open-jam mass shift at iteration 8000, but transition counts are sampled and are not yet a deterministic probability-mass proof.

## Active gate

Run a deterministic root-policy distribution audit on every forensic HU root, before any hero action is sampled.

Measure:
- TV;
- argmax disagreement;
- probability-mass change by action;
- blind and effective-stack localization.

No roots, no optimizer, no holdout.

## Immediate action

```bash
bash tools/run_lt2_hu_root_policy_drift_7600_8000.sh
```

Send `SpinCore_LT2_hu_root_policy_drift_7600_8000.json`.

Do not train beyond 8000.
