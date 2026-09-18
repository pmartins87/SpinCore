# SpinCore — Long-Training Plan

Status: **LT2 STAGE B PASS — ROOT TRAINING PAUSED — JAMMER REGRESSION LOCALIZED TO FACING ALL-IN — FINAL TARGET OVERLAY ACTIVE**
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

## Final causal overlay

Before any training, reconstruct 24 actual Jammer FACING_ALL_IN first-divergence states, balanced 4 per forensic seed.

For Stage A and Stage B separately:
- evaluate stored AveragePolicy;
- evaluate current Advantage-induced policy;
- compute stage-specific information-set posterior over opponent hands;
- build a 32-hand × 8-board conditional target reference;
- compare K1 vs K4 board-only target estimators on an independent hand stream.

Primary causal signatures:
- B-A AveragePolicy TV/regret to own reference;
- B-A Advantage TV/regret to own reference;
- FOLD/CHECK_CALL/ALL_IN shifts;
- K4-K1 estimator TV/regret;
- Stage-A vs Stage-B reference-policy shift.

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
2. Run `bash tools/run_lt2_jammer_facing_allin_target_overlay.sh`.
3. Review `SpinCore_LT2_jammer_facing_allin_target_overlay.json`.
4. Do not train K4 yet.
5. Keep holdout seeds `20261001..20261006` untouched.
