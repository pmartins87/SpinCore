# SpinCore — Long-Training Plan

Status: **LT2 STAGE B PASS — ROOT TRAINING PAUSED — FULL-POPULATION JAMMER FAI LOSS RESOLVED — STAGE-B OVERFOLDING IDENTIFIED — INFOSET CONFIRMATION ACTIVE**
Date: 2026-09-19

## Preserved milestones

Stage A:
- iteration 3000 / 1.8M roots;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Never rewrite either checkpoint.

## Resolved Jammer FAI population result

The full-population reconciliation used every HU Jammer seat-run on the forensic seeds.

Hard reproduction:
- sampled FAI contribution `-6.22672064777328`;
- exact match to the prior first-divergence result.

Deterministic expected FAI contribution:
- `-4.18434` chips/hand;
- CI95 `[-6.58924,-1.77944]`.

The FAI regression is therefore genuine in expected current-policy value.

## Stage-B overfold

At common FAI:

- Stage-A fold mass `0.24964`;
- Stage-B fold mass `0.37200`;
- difference `+0.12237`, resolved.

Contribution by fold-shift direction:

- B_MORE_FOLD:
  - `-5.57116`;
  - CI95 `[-7.61160,-3.53071]`.

- B_LESS_FOLD:
  - `+1.38682`;
  - CI95 `[+0.21539,+2.55825]`.

- NO_FOLD_SHIFT:
  - zero.

This direction-specific result supersedes the ambiguous 48-anchor broad calibration.

## Structural concentration

The largest resolved block is legal `{FOLD,CALL}`:

- contribution `-4.07764`;
- CI95 `[-5.88461,-2.27067]`.

Legal `{FOLD,CALL,ALL_IN}` is near neutral.

One-action public paths carry most of the loss.

CALL and ALL_IN have zero terminal-value spread after opponent all-in.

## Why we still do not train

The population counterfactual uses the actual hidden opponent hand and future board.

That is correct for evaluation attribution but cannot be used as the model's decision-time truth.

Before an intervention, confirm the same overfold defect using a low-noise infoset reference.

Canonical next contract:

`docs/LT2_JAMMER_FAI_FOLD_SHIFT_INFOSET_20260919.md`.

## Fold-shift infoset gate

For each forensic seed:

- collect common FAI states before hero action;
- classify only by Stage-B-minus-A fold-mass sign;
- sample 8 B_MORE_FOLD;
- sample 8 B_LESS_FOLD;
- do not condition on sampled action, terminal outcome, realized Q or low-noise Q.

Total:
- 96 anchors.

Reference:
- 64 uniform compatible opponent hands;
- 8 boards/hand;
- 512 deals/anchor.

Primary quantity:

`policy_value_B_minus_A = (sigma_B-sigma_A) dot Q_infoset`.

The matched B_LESS_FOLD group is the directional control.

## Decision logic

If B_MORE_FOLD is resolved negative under infoset Q:
- the Stage-B overfold is a genuine model/policy calibration error;
- next inspect raw Advantage margins / fallback only inside the pre-registered harmful group;
- design a minimal intervention and freeze it before touching holdout seeds.

If B_MORE_FOLD is not negative:
- do not train a fold-calibration change;
- investigate weighting / hidden-chance covariance.

No K4 training, RM loss change, or long root continuation before this gate.

## Immediate direction

1. Keep Stage A/B frozen.
2. Run `bash tools/run_lt2_jammer_fai_fold_shift_infoset.sh`.
3. Wait for `LT2_JAMMER_FAI_FOLD_SHIFT_INFOSET_PASS`.
4. Send `SpinCore_LT2_jammer_fai_fold_shift_infoset.json`.
5. Keep holdout `20261001..20261006` untouched.
6. Do not train.
