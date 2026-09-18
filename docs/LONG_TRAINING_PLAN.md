# SpinCore — Long-Training Plan

Status: **LT2 STAGE B PASS — ROOT TRAINING PAUSED — COMMON-REFERENCE V2.1 PASS — K4 TARGET BENEFIT CONFIRMED LOCALLY — FULL CAUSAL/POSTFLOP REVIEW PENDING**
Date: 2026-09-18

## Preserved milestones

Stage A:
- iteration 3000 / 1.8M roots;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Never rewrite either checkpoint.

## Strength failures

Stage B HU Jammer:
- raw `-5.141`, simultaneous CI `[-9.078,-1.204]`;
- B-A `-1.682`, CI `[-2.767,-0.597]`.

PassiveCaller HU:
- B-A `-1.261`, CI `[-2.377,-0.145]`.

Long scaling is frozen.

## Estimator findings

HU-preflop sampled-target MSE is dominated by hidden/chance variance:
- future board **65.88%**;
- opponent hand **26.10%**;
- exact-level-1 action noise **1.74%**;
- model error to conditional mean **6.27%**.

Exact1 is rejected on compute efficiency.

Board-only exact0 averaging:
- K4 is the policy-space compute elbow;
- K8 is not admitted.

K4 mechanics are verified:
- sample identity preserved;
- postflop labels unchanged;
- preflop labels changed;
- measured node multiplier `3.8771x`.

## Deployed-policy forensic result

The actual Stage-A -> Stage-B weak-baseline trajectories were paired until first hero-policy divergence.

Jammer:
- total B-A `-1.682`;
- FACING_ALL_IN contribution `-1.126`, CI excludes zero;
- root contribution `-0.557`, CI narrowly crosses zero;
- no postflop first-divergence contribution.

Therefore about two-thirds of the confirmed Jammer regression manifests specifically when responding preflop to an opponent jam.

PassiveCaller:
- largest resolved component is FLOP `-0.672`.

Uniform:
- overall unresolved;
- TURN subgroup negative and resolved.

The regression is therefore multi-mechanism. A preflop K4 intervention, even if causal for Jammer, is not sufficient evidence to resume long training globally.

## First target overlay result and correction

V1 reconstructed 24 actual Jammer FACING_ALL_IN first-divergence states.

Directionally:
- Stage B AveragePolicy TV to its own reference exceeded Stage A by `+0.0464`;
- Stage B Advantage-policy TV exceeded Stage A by `+0.0862`;
- Stage B Advantage regret exceeded Stage A by `+10.70` chips;
- K4 reduced estimator TV relative to K1 in both stages.

But V1 used stage-specific self-play posteriors for opponent hands.

That is not the correct direct benchmark reference for JAMMER. JAMMER's action rule is hand-independent, so the observed shove does not alter the hidden-hand distribution.

The V1 result is retained as training-process evidence, not final causal proof.

## Common-reference V2 assertion correction

V2 correctly used one shared Jammer-conditioned hidden-hand distribution, but its first run stopped because it asserted raw Stage-A/B Advantage-target equality.

Raw target equality is not required. At the traverser node:

`target[a] = Q(a) - V_sigma`.

Stage A and Stage B use different current `sigma`, so `V_sigma` can shift the full legal-action vector by a common scalar.

The correct invariant is the action-value geometry `Q(a)-Q(b)`.

## Common-reference V2.1 result

V2.1 passed on 24 actual Jammer FACING_ALL_IN first-divergence states.

Action-gap invariance:
- canonical Stage-A/B fixed-deal max delta `2.98e-08`;
- raw common-offset magnitude up to `0.416667`.

Policy/model summaries:
- Stage A AveragePolicy TV `0.3069`;
- Stage B AveragePolicy TV `0.3119`;
- B-A AveragePolicy regret `+0.45` chips;
- Stage A Advantage TV `0.4543`;
- Stage B Advantage TV `0.4266`;
- B-A Advantage regret `+16.73` chips.

Estimator effect:
- K4-K1 MSE `-0.026189`;
- K4-K1 TV diagnostic `-0.1146`;
- K4-K1 reference-best-action regret `-19.27` chips.

The K4 estimator is materially better on the exact Jammer-facing forensic states.

Because the common reference uses a chosen zero-mean action-gap gauge, regret/value and action gaps are primary; regret-matching TV is diagnostic only.

Forensic seeds remain `20260920..20260925`.
Untouched acceptance seeds remain `20261001..20261006`.

Do not inspect holdout seeds before an intervention is frozen.

## Decision logic

Do not authorize K4 training from the console aggregate alone.

First review the full V2.1 JSON confidence intervals and action-mass shifts.

If those confirm that Stage-B Advantage value error is resolvedly worse and K4 regret reduction is resolvedly better, K4 becomes a justified **local Jammer-facing intervention candidate**.

Before reopening long training, separately explain the resolved PassiveCaller FLOP regression and UniformLegal TURN subgroup. A broader future-chance target-variance mechanism may be preferable to a narrow HU-preflop patch.

## Immediate direction

1. Keep Stage A/B frozen.
2. Upload `SpinCore_LT2_jammer_facing_allin_common_reference_v2.json`.
3. Review confidence intervals and FOLD/CHECK_CALL/ALL_IN mass shifts.
4. Do not train K4 yet.
5. Keep holdout seeds `20261001..20261006` untouched.
