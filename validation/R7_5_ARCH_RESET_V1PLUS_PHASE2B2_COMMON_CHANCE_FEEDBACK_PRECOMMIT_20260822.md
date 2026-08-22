# R7.5 Architecture Reset — V1+ Phase2B2 Common-Chance / Feedback Decomposition

Status: **FROZEN BEFORE PHASE2B2 OUTPUTS**  
Date: 2026-08-22

## 1. Why this diagnostic exists

Phase2B1 established a strong causal asymmetry in H2 / THREE_HANDED Advantage-target generation:

- pooled K1 `TRAVERSAL_ONLY` regret-matching policy TV: `0.05194165047961052`;
- pooled K1 `CHANCE_ONLY` TV: `0.5153716032136447`;
- pooled K1 `COMBINED` TV: `0.5045434331933222`;
- chance-only / traversal-only ratio: about `9.92x`.

Therefore deck/hidden/future chance realization is the dominant upstream source under the Phase2B1 design. However, Phase2B1 did **not** answer a second causal question: if the two already-diverged source behavior ensembles are evaluated on the *same exact chance support*, do their Advantage targets become similar, or does policy-feedback divergence remain large even on identical deals?

Phase2B2 answers that question before any new model fit or training run.

## 2. Governance

- R7.5.3 remains `FAIL_BLOCKED_CLOSED`.
- Phase2B0 candidate remains `FAIL_DO_NOT_TRAIN_CANDIDATE`.
- Phase2B1 remains `PHASE2B1_K4_SCREEN_FAIL_NO_GENERIC_K4_TRAINING_PILOT`.
- Phase2B2 is read-only with respect to all learned state: no optimizer step, no model fit, no reservoir insertion, no checkpoint mutation.
- Fresh solver traversal is permitted only for causal measurement.
- A Phase2B2 pass may authorize freezing **one small shared-chance-support training pilot**. It does not authorize production training or table use.
- `READY FOR TABLES = NO`.

## 3. Frozen source objects

Representation: `H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL`  
Domain: `THREE_HANDED`  
Action candidate: `PF0_CONTROL_33_75_AI`  
Exact opponent levels: `2`  
Target iteration tag: `3`  
Source Phase2A execution SHA: `4bfa55d69029cd69536fa6dbfcadd162719cb887`  
Frozen source behavior seeds: `1342191342`, `1801739323`  
Ensemble members per source behavior: `4`

Phase2B2 consumes the completed local Phase2B1 result and requires its status to be exactly `PHASE2B1_K4_SCREEN_FAIL_NO_GENERIC_K4_TRAINING_PILOT` with source classification `CHANCE_DOMINANT`.

## 4. Chance support

Use exactly the 15 Phase2B1 THREE_HANDED collision groups and exactly their 16 stored deck seeds per scenario.

For every group, reconstruct the root from every stored deck seed and require:

- exact root SPNNIV3 SHA-256 equal to the Phase2B1 recorded observation SHA-256;
- same actor;
- same legal universal action set;
- same scenario descriptor.

Thus both frozen behavior ensembles are evaluated on exactly the same observable root infoset and the same hidden/future chance realizations.

No new collision search is allowed in Phase2B2.

## 5. Two traversal-RNG coupling arms

For each scenario and each of the 16 paired deck seeds, generate one root Advantage target under each frozen behavior ensemble.

### A. `COMMON_TRAVERSAL_RNG`

The two behavior ensembles receive exactly the same deterministic traversal RNG seed on the same deal. This is a common-random-numbers coupling. It does not force identical actions: different behavior probabilities may map the same random draw to different actions.

### B. `INDEPENDENT_TRAVERSAL_RNG`

The two behavior ensembles receive deterministic but different traversal RNG namespaces on the same deal. This prevents a favorable common-random-number coupling from being mistaken for a true same-chance-support effect.

Because Phase2B1 already showed traversal-only variance is small, `INDEPENDENT_TRAVERSAL_RNG` is the primary decision arm; `COMMON_TRAVERSAL_RNG` is a variance-reduced diagnostic cross-check.

## 6. Paired target comparison

Each target is the exact root Advantage sample for `traverser=root.actor`, extracted using exact observation and legal-mask identity.

For each scenario/arm, pair behavior A and behavior B on the same deck realization. For `K in {1,2,4,8,16}`:

- partition the 16 paired deals into deterministic non-overlapping blocks of size K;
- average raw ten-slot Advantage targets separately for behavior A and B over the same block;
- compare the two block means on legal slots;
- convert each block mean to the existing regret-matching policy only for diagnostics.

Record:

- legal-slot target mean absolute difference;
- positive/non-positive sign disagreement fraction;
- regret-matching policy TV;
- dominant legal action mismatch.

Pooled comparison counts per arm are therefore 240, 120, 60, 30 and 15 for K=1/2/4/8/16.

## 7. Frozen decision thresholds

Reference chance variance is the Phase2B1 pooled K1 `CHANCE_ONLY` policy TV:

`REFERENCE_CHANCE_K1_TV = 0.5153716032136447`.

### Shared-chance-support effect

The same-chance-support effect is **materially supported** only if the primary `INDEPENDENT_TRAVERSAL_RNG` K1 cross-behavior policy TV:

1. is at least `0.10` absolute below the reference chance K1 TV;
2. is at least `30%` relatively below the reference chance K1 TV;
3. is `<= 0.35` absolute;
4. and the `COMMON_TRAVERSAL_RNG` K1 TV is not more than `0.05` worse than the independent arm.

If all four hold, Phase2B2 may route to `PRECOMMIT_SMALL_SHARED_CHANCE_SUPPORT_TRAINING_PILOT`.

### Feedback-dominant result

If `INDEPENDENT_TRAVERSAL_RNG` K1 TV is at least `80%` of the Phase2B1 chance reference (`>= 0.4122972825709158`), then changing chance support alone is not sufficient and the next route is `DIAGNOSE_BEHAVIOR_FEEDBACK_STABILIZATION_BEFORE_TRAINING`.

### Intermediate result

Otherwise the result is `MIXED_CHANCE_SUPPORT_AND_FEEDBACK`; no training pilot is authorized yet. The next diagnostic must localize the remaining feedback divergence by scenario/action/street or iteration mechanism.

### K aggregation is secondary only

K=2/4/8/16 is used to characterize convergence under *paired common chance support*. It does not revive the rejected generic K4 training candidate. A favorable K curve may motivate later stratified/common-support engineering, but cannot by itself override a failed K1 shared-support gate.

## 8. Compute contract

- 15 scenarios × 16 deck replicates × 2 source behaviors × 2 RNG-coupling arms = 960 root Advantage traversals.
- Up to 12 independent worker processes.
- One Torch/OMP/MKL thread per worker.
- Each worker loads both frozen four-member behavior ensembles once.
- Aggregation is sorted by `(scenario_index, arm)` and is independent of process completion order.

## 9. Interpretation guardrail

A large reduction under identical chance support would establish that different deck support is a major driver of cross-seed target divergence, but it would **not** prove that simply giving every production learner the same finite deck list generalizes. Any later shared-support pilot must retain an independent chance-block validation before architecture admission or strategic evaluation.
