# SpinCore — LT2 Jammer FAI raw-margin decomposition result

Date: 2026-09-19  
Status: **PASS — ACTION-GAP DRIFT IS THE LARGER CAUSAL COMPONENT; COMMON OFFSET/FALLBACK ALSO CONTRIBUTES; FALLBACK-ONLY PATCH IS INSUFFICIENT**

## Source

Derived read-only from the completed 384-anchor structural infoset confirmation.

No solver, training roots, optimizer steps, memory writes, or holdout seeds were used.

For legal `{FOLD,CALL}`, raw Advantage outputs were decomposed into:

```
center = (raw_fold + raw_continue) / 2
gap    = raw_fold - raw_continue
```

The exact production lean regret-matching map was then applied to counterfactual hybrids.

## B_MORE_FOLD — primary cohort

Full observed Stage-B-minus-A policy value:

- `-24.549` chips;
- seed-cluster CI95 `[-35.584,-13.514]`.

### GAP_ONLY

Stage-A center + Stage-B fold-vs-continue gap:

- `-18.719` chips;
- CI95 `[-27.359,-10.080]`.

This is resolved negative and is the largest decomposed component.

Numerically it is about 76% of the full mean loss.

### OFFSET_ONLY

Stage-B center + Stage-A gap:

- `-4.946` chips;
- CI95 `[-9.166,-0.726]`.

This is also resolved negative.

Common-offset / zero-crossing drift therefore contributes materially, but much less than action-gap drift.

### Nonlinear interaction

- `-0.884` chips;
- CI95 `[-7.942,+6.173]`.

Unresolved.

The full effect is therefore well explained by a mixed mechanism dominated by fold-vs-continue gap drift plus a smaller common-offset component.

## Fallback transitions

Within 192 B_MORE_FOLD anchors:

- fallback -> fallback: 11;
- fallback -> nonfallback: 1;
- nonfallback -> fallback: 85;
- nonfallback -> nonfallback: 95.

The 85 nonfallback->fallback anchors have mean full B-A approximately `-42.55` chips and account for about 76.7% of the net cohort loss.

This confirms that entering fallback is a major downstream amplifier.

However, transition status is downstream of both center and gap movement; it must not be mistaken for the sole root cause.

## Diagnostic-only argmax fallback probe

Replace only the Stage-B all-nonpositive softmax fallback with deterministic raw argmax; keep production regret matching otherwise unchanged.

Recovery versus production Stage B:

- `+15.415` chips;
- CI95 `[+8.623,+22.207]`.

The modified Stage-B-vs-A value becomes:

- `-9.134` chips;
- CI95 `[-20.751,+2.483]`.

Thus a fallback-only patch recovers a large part of the loss but does **not** establish parity with Stage A and does not repair the resolved action-gap degradation.

No fallback change is authorized from this probe.

## B_LESS_FOLD directional control

Full B-A:

- `+3.053`;
- CI95 `[-5.042,+11.148]`.

OFFSET_ONLY:

- `+4.407`;
- CI95 `[-0.040,+8.855]`.

GAP_ONLY:

- `-1.413`;
- CI95 `[-6.196,+3.369]`.

No primary control contrast resolves.

## Verdict

The next target is **not** a one-line fallback rewrite.

The confirmed Stage-B FAI regression is driven primarily by a worse fold-vs-continue Advantage gap, with a smaller but real common-offset / fallback contribution.

The next gate must distinguish:

1. Stage-B reservoir / target-signal deterioration;
2. instability or underfitting of the particular last Advantage refit.

The correct next experiment is a controlled fresh-refit audit using the Stage-A and Stage-B HU Advantage reservoirs with identical initialization and batch-sampling seeds on the same fixed 384-anchor low-noise cohort.
