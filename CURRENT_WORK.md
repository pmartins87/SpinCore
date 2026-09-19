# SpinCore Current Work

Date: 2026-09-19
Status: **LT2 STAGE B PASS — JAMMER FAI INFOSET OVERFOLD CONFIRMED — RAW MARGIN DECOMPOSITION NEXT — NO TRAINING**

## Active source of truth

Read before new compute:

- `docs/LT2_JAMMER_FAI_STRUCTURAL_INFOSET_RESULT_20260919.md`
- `docs/LT2_JAMMER_FAI_RAW_MARGIN_DECOMPOSITION_20260919.md`
- `docs/LT2_JAMMER_FAI_POPULATION_RECONCILIATION_RESULT_20260919.md`
- `docs/LONG_TRAINING_PLAN.md`

Preserve Stage A and Stage B. Do not continue root training beyond iteration 7500.

## Preserved checkpoints

Stage A:
- iteration 3000 / 1.8M roots;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Powered structural confirmation — decisive PASS

Frozen structure:

- common Jammer FAI;
- legal exactly `{FOLD,CALL}`;
- exactly one public action before FAI;
- 192 B_MORE_FOLD + 192 B_LESS_FOLD anchors;
- 512 explicit low-noise reference deals per anchor;
- holdout untouched.

### Primary B_MORE_FOLD result

Policy-value B-A:

- `-24.54921` chips;
- seed-cluster CI95 `[-35.58425,-13.51416]`.

Fold-mass B-A:

- `+0.46907`;
- CI95 `[+0.45307,+0.48508]`.

Reference FOLD-minus-CONTINUE:

- `-40.86355` chips;
- CI95 `[-57.51783,-24.20927]`.

The Stage-B overfold is therefore confirmed at the decision-time infoset level. It is not explained by realized hidden-card / future-board covariance.

### Model geometry

Canonical action-gap MSE B-A:

- `+0.000714714`;
- CI95 `[+0.000397781,+0.001031647]`.

Class-error mass B-A:

- `+0.127198`;
- CI95 `[+0.058510,+0.195887]`.

Raw-target MSE does not resolve.

Fallback rate:

- Stage A `6.25%`;
- Stage B `50.00%`.

This is a mixed signal: fallback/zero-crossing is suspicious, but the fold-vs-continue action gap itself also degrades.

## Active gate

Run a read-only raw-margin decomposition on the already completed structural report.

For each anchor, decompose the two legal raw outputs into:

- common center / offset;
- fold-vs-continue gap.

Recombine Stage-A/Stage-B center and gap under exact production lean regret matching.

Also run one diagnostic-only argmax fallback counterfactual.

No solver, roots, optimizer steps, memory writes, or holdout use.

## Immediate user action

Pull `main` and run:

```bash
bash tools/run_lt2_jammer_fai_raw_margin_decomposition.sh
```

Send `SpinCore_LT2_jammer_fai_raw_margin_decomposition.json`.

Do not start K4 training or resume long training.
