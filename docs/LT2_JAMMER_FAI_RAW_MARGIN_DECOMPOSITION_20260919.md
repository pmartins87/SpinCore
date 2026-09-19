# SpinCore — LT2 Jammer FAI raw-margin decomposition

Date: 2026-09-19  
Status: **ACTIVE — READ-ONLY DECOMPOSITION OF THE CONFIRMED STRUCTURAL OVERFOLD**

## Trigger

The powered structural infoset confirmation resolved:

- B_MORE_FOLD policy value B-A `-24.54921` chips;
- seed-cluster CI95 `[-35.58425,-13.51416]`;
- fold-mass shift `+0.46907`;
- canonical action-gap MSE degradation `+0.000714714`, resolved;
- class-error mass degradation `+0.127198`, resolved;
- fallback incidence Stage A `6.25%` -> Stage B `50.00%`.

Raw-target MSE did not resolve.

Therefore the next question is not whether the overfold exists. It is which part of the raw Advantage change creates it.

## Two-action decomposition

Inside the frozen structure legal actions are exactly FOLD and CALL.

For each stage define:

```
center = (raw_fold + raw_continue) / 2
gap    = raw_fold - raw_continue
```

Reconstruct three policies using the exact production lean regret-matching mapping:

- A: Stage-A center + Stage-A gap;
- OFFSET_ONLY: Stage-B center + Stage-A gap;
- GAP_ONLY: Stage-A center + Stage-B gap;
- FULL_B: Stage-B center + Stage-B gap.

This separates common-offset/zero-crossing effects from fold-vs-continue action-gap drift without creating new targets or touching training.

## Bounded fallback probe

Also evaluate one diagnostic-only counterfactual:

- retain production regret matching whenever any legal raw Advantage is positive;
- if every legal raw Advantage is non-positive, choose the raw argmax instead of the current softmax fallback.

This is **not** an authorized runtime change. It only measures how much of the confirmed loss can possibly be recovered by changing fallback semantics alone.

## Inputs

Use the already completed structural confirmation JSON.

No solver is required.

No checkpoints are mutated.

No training roots, optimizer steps, memory writes, or holdout seeds are touched.

## Decision logic

If GAP_ONLY carries most of the resolved loss:
- prioritize Advantage-gap fit / training-data diagnostics;
- do not patch fallback alone.

If OFFSET_ONLY carries most of the loss:
- prioritize common-offset / zero-crossing stabilization and fallback semantics.

If both resolve:
- treat the defect as mixed and quantify whether a fallback-only patch leaves material residual loss.

If the argmax-fallback probe materially recovers value but FULL_B remains worse than A:
- fallback is a contributor, not the root cause;
- next test must distinguish reservoir/target-signal drift from last-fit instability.

Holdout `20261001..20261006` remains sealed.
