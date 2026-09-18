# SpinCore — LT2 Jammer facing-all-in target overlay

Date: 2026-09-18
Status: **ACTIVE — FINAL CAUSAL OVERLAY BEFORE ANY K4 TRAINING**

## Trigger

The completed Stage-A -> Stage-B deployed-policy forensic found:

- HU Jammer B-A `-1.682` chips/hand, 95% CI `[-2.767,-0.597]`;
- only `4.8%` of Jammer seat-runs first-diverge at all;
- `PREFLOP_FACING_ALL_IN` occurs in only `2.5%` of Jammer seat-runs but contributes `-1.126` chips/hand;
- 95% CI for that contribution is `[-2.049,-0.202]`;
- this one class explains about **67%** of the total Jammer Stage-B regression.

This independently localizes the deployed-policy failure to the same broad state class in which earlier target diagnostics found large hidden/future-board noise.

However, the forensic also found a separate resolved PassiveCaller FLOP regression and a UNIFORM_LEGAL TURN subgroup. K4 is therefore not assumed to be a universal fix.

Canonical forensic result:

- `docs/LT2_STAGE_A_B_FIRST_DIVERGENCE_RESULT_20260918.md`

## Remaining causal question

The missing causal chain is:

`noisy Advantage target -> wrong current Advantage policy -> wrong accumulated AveragePolicy -> negative Jammer EV`.

The current overlay tests that chain on **actual evaluation failure states**, not on generic self-play anchors.

## State selection

Use only the already-seen forensic seed family:

- 20260920
- 20260921
- 20260922
- 20260923
- 20260924
- 20260925

Reconstruct the same HU Jammer evaluation trajectories.

Select a deterministic balanced sample of 4 states per seed = 24 total where:

1. Stage A and Stage B are still on an identical solver state;
2. their sampled hero actions first diverge;
3. the state is preflop;
4. the immediately preceding opponent action is ALL_IN.

The reserved acceptance family `20261001..20261006` remains untouched.

## Lightweight source extraction

The full Stage-A/B checkpoints are read only long enough to extract, per HU domain:

- AveragePolicy weights;
- Advantage weights;
- Advantage-ready flag;
- domain seed;
- completed iteration.

The large reservoirs are not retained.

## Stage-specific conditional reference

For each forensic state and each stage separately:

1. preserve the observable information state and hero cards;
2. enumerate all 2,450 ordered opponent hands;
3. weight opponent hands by that stage's current self-play behavior reach for the observed public path;
4. select 32 posterior-stratified opponent-hand draws;
5. sample 8 future boards per selected hand;
6. generate exact-level-0 targets;
7. average the 256 target realizations.

At these anchors the opponent is already all-in, so no future opponent action remains. Exact-opponent branching has nothing to integrate; hidden opponent hand and future board are the relevant remaining uncertainty.

The reference is still a current-target-process conditional estimate, not a GTO oracle.

## Overlay metrics

Against each stage's own low-noise conditional reference:

### Stored AveragePolicy
- policy TV;
- argmax agreement;
- reference-value gap/regret;
- FOLD / CHECK_CALL / ALL_IN mass.

### Current Advantage model
- raw prediction -> regret-matching policy;
- policy TV;
- argmax;
- reference-value gap/regret;
- FOLD / CHECK_CALL / ALL_IN mass.

### Target estimator
Using an independent posterior-hand stream:
- 16 sampled opponent hands;
- 4 future boards per fixed hand;
- compare K1 against K4 board averaging;
- MSE to reference;
- policy TV;
- argmax;
- reference-value gap/regret;
- node cost.

Also report:
- Stage-B minus Stage-A AveragePolicy error;
- Stage-B minus Stage-A Advantage-policy error;
- Stage-A vs Stage-B reference-policy TV;
- actual sampled A->B action-transition counts;
- forensic terminal B-A delta for selected states.

## Decision logic

Evidence supports the K4 causal hypothesis only if the same states show a coherent chain such as:

1. Stage-B AveragePolicy is farther from its own conditional target than Stage A;
2. Stage-B current Advantage policy is also farther from target in the same action direction;
3. the FOLD/CHECK_CALL/ALL_IN shift matches the deployed-policy regression;
4. K4 reduces target-estimator error versus K1 on those exact states.

If AveragePolicy worsens but current Advantage does not, suspect AveragePolicy/reservoir accumulation rather than Advantage-target noise.

If Advantage worsens but K4 does not improve target estimates on the forensic states, reject K4 as the causal fix.

Even a positive result authorizes only a bounded K4 pilot for the Jammer-facing preflop mechanism. It does not resolve the separate postflop regression.

## Launcher

```bash
bash tools/run_lt2_jammer_facing_allin_target_overlay.sh
```

Expected marker:

`LT2_JAMMER_FACING_ALLIN_TARGET_OVERLAY_PASS`

Do not train K4 before this overlay is reviewed.
