# SpinCore — Long-Training Plan

Status: **LT2 STAGE B PASS — ROOT TRAINING PAUSED — JAMMER FAI INFOSET OVERFOLD CONFIRMED — RAW MARGIN DECOMPOSITION ACTIVE**
Date: 2026-09-19

## Preserved milestones

Stage A:
- iteration 3000 / 1.8M roots;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Never rewrite either checkpoint.

## What is now resolved

The natural Jammer population first established:

- deterministic expected FAI B-A `-4.18434` chips/hand;
- CI95 `[-6.58924,-1.77944]`;
- Stage-B fold mass +12.24 pp;
- B_MORE_FOLD harmful;
- B_LESS_FOLD beneficial.

The powered low-noise infoset confirmation then froze the upstream high-impact structure:

- legal `{FOLD,CALL}`;
- one public action before FAI.

Within 192 B_MORE_FOLD anchors:

- policy-value B-A `-24.54921`;
- CI95 `[-35.58425,-13.51416]`;
- fold-mass B-A `+0.46907`;
- reference FOLD-minus-CONTINUE `-40.86355` chips;
- canonical action-gap MSE worsens by `+0.000714714`, resolved;
- class-error mass worsens by `+0.127198`, resolved;
- fallback incidence rises from `6.25%` to `50.00%`;
- raw-target MSE does not resolve.

Therefore the Stage-B overfold is confirmed at infoset level and is not a hidden-card / future-board covariance artifact.

## Why training is still paused

The defect is confirmed, but the smallest safe intervention is not yet identified.

Two mechanisms can coexist:

1. common raw-output offset crosses the zero boundary and changes fallback behavior;
2. fold-vs-continue raw action gap itself degrades.

Changing fallback prematurely could mask, but not repair, a real action-gap fit problem.

No K4 training, fallback patch, regret-matching rewrite, or long-root continuation is authorized yet.

## Active diagnostic

Run the raw-margin decomposition from the completed structural JSON.

No new poker simulation is required.

For each anchor:

- decompose Stage A/B legal raw outputs into center and gap;
- recombine center/gap counterfactually;
- pass each hybrid through exact production lean regret matching;
- compute low-noise policy value using the already stored Q reference;
- evaluate a diagnostic-only argmax fallback probe.

## Decision logic after decomposition

If action-gap drift is dominant:
- next compare Stage-A vs Stage-B Advantage reservoirs under controlled fresh refits on the same fixed structural cohort;
- distinguish training-data/target-signal drift from last-fit instability.

If common-offset/fallback drift is dominant:
- design a translation/zero-crossing stabilization candidate;
- evaluate it on forensic data before opening holdout.

If both matter:
- do not accept a one-line fallback fix unless it removes the material residual gap error too.

Only after an intervention is frozen may holdout `20261001..20261006` be opened.

## Immediate direction

1. Keep Stage A/B frozen.
2. Pull `main`.
3. Run `bash tools/run_lt2_jammer_fai_raw_margin_decomposition.sh`.
4. Send `SpinCore_LT2_jammer_fai_raw_margin_decomposition.json`.
5. Do not train.
