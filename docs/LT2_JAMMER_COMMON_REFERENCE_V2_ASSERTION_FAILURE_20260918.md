# SpinCore — Common-reference V2 first-run assertion failure

Date: 2026-09-18
Status: **DIAGNOSED — ASSERTION WAS TOO STRONG; ACTION-GAP GAUGE FIXED IN V2.1**

## Observed failure

The first common-reference V2 stopped on the first anchor with:

`RuntimeError: Stage-A/B fixed-deal target differs after opponent already all-in: max_abs=0.014183718711137772`.

No training occurred and both preserved checkpoints remained read-only.

## Why raw targets can differ even with identical fixed-deal action values

The Deep-CFR collector stores the traverser's Advantage target as:

`target[a] = Q(a) - V_sigma`

with:

`V_sigma = sum_b sigma(b) Q(b)`.

Stage A and Stage B have different current Advantage policies `sigma`.

Therefore, even when:
- the solver state is identical;
- opponent hand is identical;
- future board is identical;
- opponent is already all-in;
- all action values `Q(a)` are identical across stages;

the raw Advantage vectors can differ by one **common scalar offset** because `V_sigma` differs.

The V2 equality assertion incorrectly required the raw centered labels themselves to be identical.

## Correct invariant

What must be stage-invariant is the **action-value geometry**:

`Q(a) - Q(b)`

for every pair of legal actions.

Equivalently, choose one fixed gauge:

`canonical[a] = target[a] - mean_legal(target)`

which equals:

`Q(a) - mean_legal(Q)`.

This removes the stage-specific `V_sigma` baseline while preserving all action gaps.

## V2.1 correction

The common-reference audit now:

1. computes raw Stage-A and Stage-B targets for the same explicit deal;
2. canonicalizes both by subtracting the equal-weight mean over legal actions;
3. requires canonical targets to match to numerical tolerance;
4. separately requires the raw A-B difference across legal actions to be constant;
5. records the magnitude of the raw common offset rather than treating it as an error;
6. builds the common Jammer reference from canonical targets;
7. canonicalizes K1/K4 target samples before estimator comparison.

AveragePolicy and current Advantage-induced policies are then evaluated against this single common action-value reference.

## Scientific meaning

This is not a new strategy failure.

It exposes an important property of the existing training target: its absolute zero is policy-dependent.

For cross-stage causal comparison, action gaps / a fixed canonical gauge are the correct invariant.

The fixed Jammer hidden-hand posterior correction from V2 remains valid.

## Next action

Rerun the same launcher:

```bash
bash tools/run_lt2_jammer_facing_allin_common_reference_v2.sh
```

Expected final marker:

`LT2_JAMMER_FACING_ALLIN_COMMON_REFERENCE_V2_1_PASS`

No K4 training is authorized before V2.1 is reviewed.
