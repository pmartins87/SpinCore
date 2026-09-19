# SpinCore — LT2 Jammer FAI fold-shift low-noise infoset gate

Date: 2026-09-19
Status: **ACTIVE — TEST WHETHER THE RESOLVED STAGE-B OVERFOLD LOSS PERSISTS UNDER INFOSET EXPECTATION**

## Trigger

The full-population actual-deal reconciliation resolved:

- expected FAI B-A `-4.18434` chips/hand;
- CI95 `[-6.58924,-1.77944]`;
- Stage-B fold mass +12.24 pp;
- B_MORE_FOLD contribution `-5.57116`, resolved;
- B_LESS_FOLD contribution `+1.38682`, resolved beneficial;
- no-fold-shift contribution zero.

The sampled prior FAI result `-6.22672064777328` was reproduced exactly.

Therefore the FAI loss is genuine in expected evaluation value.

## Remaining causal question

The actual-deal estimator uses the already dealt opponent hand and full board.

That is valid for population attribution, but it is not an infoset target for the model.

We now need to know whether the harmful Stage-B fold increase is also wrong under the information actually available at decision time.

## Pre-reference selection

For every forensic seed:

1. collect common Jammer FAI states before the FAI hero action;
2. compute production Stage-A and Stage-B fold mass;
3. classify only by:
   - `B_MORE_FOLD`;
   - `B_LESS_FOLD`;
   - `NO_FOLD_SHIFT`;
4. uniformly sample:
   - 8 B_MORE_FOLD;
   - 8 B_LESS_FOLD.

No state is selected using:

- sampled FAI action;
- terminal B-A outcome;
- actual dealt action value;
- low-noise Q reference;
- optimal class.

This protects the next diagnostic from direct outcome cherry-picking.

## Reference

For each selected anchor:

- opponent hand posterior is uniform because JAMMER is hand-independent;
- 64 uniformly stratified compatible opponent hands;
- 8 future boards per hand;
- 512 explicit deals per anchor;
- exact0 after opponent is already all-in;
- common canonical action-gap gauge.

Total:
- 96 anchors;
- 49,152 explicit reference deals.

## Primary metrics

Per shift group:

- Stage-B-minus-A policy value in chips:
  `(sigma_B-sigma_A) dot Q_infoset`;
- Stage-B-minus-A policy regret;
- reference FOLD-minus-CONTINUE value;
- Stage-B-minus-A class-error mass;
- Stage-B-minus-A fold mass;
- canonical action-gap MSE difference;
- raw-target MSE difference.

CALL and ALL_IN are required to be value-equivalent to numerical tolerance.

## Decision logic

### If B_MORE_FOLD has resolved negative infoset policy value

Then the overfold is a real model/policy error under the information set.

Next step:
- inspect raw Advantage fold-vs-continue margins / fallback regimes inside this pre-registered group;
- design the smallest intervention that fixes fold calibration without perturbing other streets.

### If B_MORE_FOLD is neutral/positive under infoset Q

Then actual-deal population harm does not translate to the decision-time expectation.

Do not train a fold-calibration intervention; investigate weighting / hidden-chance covariance instead.

### Control

B_LESS_FOLD is a matched policy-shift control.

If it remains positive under infoset Q while B_MORE_FOLD is negative, the direction-specific fold-calibration mechanism is strongly supported.

## Launcher

```bash
bash tools/run_lt2_jammer_fai_fold_shift_infoset.sh
```

Expected marker:

`LT2_JAMMER_FAI_FOLD_SHIFT_INFOSET_PASS`.

Holdout `20261001..20261006` remains untouched.

No training is authorized.
