# SpinCore — Long-Training Plan

Status: **LT2 STAGE B PASS — ROOT TRAINING PAUSED — JAMMER REGRESSION LOCALIZED — COMMON REFERENCE RETAINED — ACTION-GAP V2.1 ACTIVE**
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

## Corrected final causal overlay — V2.1

V2.1 keeps:
- uniform compatible Jammer opponent hands;
- uniform future boards;
- 32 hands × 8 boards;
- one shared benchmark reference.

It now canonicalizes each target to:

`Q(a)-mean_legal(Q)`

by subtracting the legal-action mean.

It asserts:
- canonical Stage-A/B fixed-deal targets match;
- any raw Stage-A/B difference is constant across legal actions.

Primary causal signatures:
- B-A AveragePolicy TV/regret to the same reference;
- B-A Advantage TV/regret to the same reference;
- FOLD/CHECK_CALL/ALL_IN shifts;
- K4-K1 estimator TV/regret against that same reference.

Forensic seeds:
`20260920..20260925`.

Untouched acceptance seeds:
`20261001..20261006`.

Do not inspect holdout seeds before an intervention is frozen.

## Decision logic

A bounded K4 pilot is admissible only if:
- Stage B is worse than Stage A on the actual Jammer-facing failure states;
- current Advantage degradation aligns with AveragePolicy degradation;
- lower-variance target reference points in the corrective direction;
- K4 improves target estimation on those states.

If not, investigate AveragePolicy/reservoir dynamics instead.

Even a positive K4 result does not reopen long training until the separate postflop regression is addressed.

## Immediate direction

1. Keep Stage A/B frozen.
2. Run `bash tools/run_lt2_jammer_facing_allin_common_reference_v2.sh`.
3. Wait for `LT2_JAMMER_FACING_ALLIN_COMMON_REFERENCE_V2_1_PASS`.
4. Review `SpinCore_LT2_jammer_facing_allin_common_reference_v2.json`.
4. Do not train K4 yet.
5. Keep holdout seeds `20261001..20261006` untouched.
