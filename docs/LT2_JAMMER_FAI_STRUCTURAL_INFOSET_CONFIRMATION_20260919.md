# SpinCore — LT2 Jammer FAI structural fold-shift infoset confirmation

Date: 2026-09-19
Status: **ACTIVE — POWERED CONFIRMATION ON THE PRE-EXISTING HIGH-IMPACT FAI STRUCTURE**

## Trigger

The first fold-shift low-noise infoset audit produced directionally aligned but unresolved results:

- B_MORE_FOLD policy B-A `-5.92295`, CI95 `[-19.60209,+7.75619]`;
- B_LESS_FOLD policy B-A `+7.47850`, CI95 `[-7.99668,+22.95369]`.

The audit was valid and holdout-clean, but 48 anchors/group were not enough for the heterogeneous full FAI population.

## Why structural focus is allowed

Before the first infoset audit, the full-population reconciliation had already established that most deterministic expected loss was concentrated in:

- legal actions `{FOLD,CALL}`:
  `-4.07764` chips/hand, resolved;
- one public action before FAI:
  `-3.52854`, resolved.

CALL and ALL_IN are terminal-value equivalent after the Jammer is all-in.

Therefore focusing on the intersection:

`legal == {FOLD,CALL} AND common_public_action_count == 1`

is based on an upstream, documented localization rather than on the current low-noise reference outcomes.

## Selection

For each forensic seed:

1. collect every common Jammer FAI state;
2. require:
   - legal actions exactly slots `0,1`;
   - one public action before FAI;
3. classify only from Stage-B-minus-A production fold mass:
   - B_MORE_FOLD;
   - B_LESS_FOLD;
4. uniformly sample:
   - 32 B_MORE_FOLD;
   - 32 B_LESS_FOLD.

Total:

- 192 B_MORE_FOLD;
- 192 B_LESS_FOLD;
- 384 anchors.

No state is selected using:

- sampled FAI action;
- terminal outcome;
- actual-deal action value;
- low-noise reference;
- optimal class.

## Reference

Per anchor:

- Jammer hand-independent -> uniform compatible opponent hands;
- 64 uniformly stratified hands;
- 8 future boards/hand;
- 512 explicit deals;
- common canonical action-gap gauge.

Total planned reference deals:

`384 × 512 = 196,608`.

## Primary metric

For each shift group:

`policy_value_B_minus_A = (sigma_B-sigma_A) dot Q_infoset`.

Primary confirmatory condition:

- B_MORE_FOLD must have a resolved negative seed-cluster CI.

Directional control:

- B_LESS_FOLD is expected to be positive if the same fold-direction mechanism is present.

B_LESS_FOLD is supportive control, not required to authorize the next diagnostic if the primary B_MORE_FOLD condition resolves.

## Decision logic

### If B_MORE_FOLD resolves negative

The Stage-B overfold defect is confirmed at decision-time infoset level inside the pre-existing high-impact FAI structure.

Next:
- inspect raw Advantage fold-vs-continue margins, fallback incidence, and target-estimator noise specifically in this frozen structure;
- identify the smallest training-side cause/intervention;
- still do not touch holdout until an intervention is frozen.

### If B_MORE_FOLD remains unresolved

Do not keep escalating sample size blindly.

Then quantify whether the remaining variance is:
- state heterogeneity;
- low-noise reference Monte Carlo noise;
- or seed-level composition.

Choose the estimator/intervention path only after that variance decomposition.

### If B_MORE_FOLD resolves positive

The infoset reference contradicts the actual-deal attribution in this structural block.

Do not train a fold fix; investigate evaluation weighting / hidden-chance covariance.

## Launcher

```bash
bash tools/run_lt2_jammer_fai_structural_infoset_confirmation.sh
```

Expected marker:

`LT2_JAMMER_FAI_STRUCTURAL_INFOSET_CONFIRMATION_PASS`.

Holdout `20261001..20261006` remains untouched.

No training is authorized.
