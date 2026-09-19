# SpinCore Roadmap — active state 2026-09-19

## Active status

- LT0 — **DONE**.
- LT1 — **DONE**.
- LT2 Stage A — **PASS**.
- LT2 Stage B — **PASS** at 4.5M roots / iteration 7500.
- Weak-baseline regression — **CONFIRMED**.
- AveragePolicy extra-fit — **NOT SUPPORTED**.
- Advantage optimizer escalation — **NOT SUPPORTED**.
- HU-preflop target variance — **HIDDEN/CHANCE DOMINANT**.
- K4 board averaging — **VALID ESTIMATOR IMPROVEMENT**.
- Jammer K4 causal gate — **NOT MET**.
- Cross-street future-chance audit — **PASS; FUTURE-CHANCE NOT FAILURE-SPECIFIC**.
- Stage-A/B target-drift matrix — **PASS; NO UNIVERSAL TARGET-DRIFT EXPLANATION**.
- HU policy-chain audit — **PASS; JAMMER DEFECT UPSTREAM IN CURRENT BEHAVIOR**.
- HU current-behavior first divergence — **PASS; 73.86% OF JAMMER LOSS AT PREFLOP FACING ALL-IN**.
- Broad Jammer FAI calibration — **PASS; SMALL 48-ANCHOR SAMPLE DID NOT SHOW BROAD B DEGRADATION**.
- Full-population Jammer FAI reconciliation — **PASS; EXPECTED FAI LOSS RESOLVED AND CARRIED BY B OVERFOLDING**.
- Fold-shift low-noise infoset audit — **PASS / UNDERPOWERED; DIRECTIONS ALIGN BUT PRIMARY CIs CROSS ZERO**.
- Powered structural infoset confirmation — **NEXT**.
- K4 training — **NOT AUTHORIZED**.
- long root training — **PAUSED**.
- DeepCrusher — **DEFERRED**.

## Preserved checkpoints

Stage A SHA:
`e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B SHA:
`3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Canonical full-population result

Deterministic expected FAI contribution:

- `-4.18434` chips/hand;
- CI95 `[-6.58924,-1.77944]`.

B_MORE_FOLD:

- contribution `-5.57116`;
- resolved harmful.

B_LESS_FOLD:

- contribution `+1.38682`;
- resolved beneficial.

## First low-noise fold-shift infoset result

### B_MORE_FOLD

Policy-value B-A:

- `-5.92295`;
- seed-cluster CI95 `[-19.60209,+7.75619]`.

### B_LESS_FOLD

Policy-value B-A:

- `+7.47850`;
- seed-cluster CI95 `[-7.99668,+22.95369]`.

The signs match the full-population result, but 48 anchors/group do not resolve the infoset effect.

No canonical action-gap MSE or class-error degradation resolves in B_MORE_FOLD.

Verdict:

**inconclusive due power / heterogeneity**.

Do not infer that the mechanism is absent.

## Pre-existing structural localization

The full-population result had already identified:

- legal slots `0,1` as the dominant resolved loss block;
- one public action before FAI as the dominant path-length block.

Therefore the next confirmation freezes that structure before reference evaluation.

## Next gate

Run:

`tools/run_lt2_jammer_fai_structural_infoset_confirmation.sh`.

Selection:

- common Jammer FAI;
- legal exactly `0,1`;
- path length exactly 1;
- classify only by fold-mass shift sign;
- no action/outcome/Q-based selection.

Sample:

- 32 B_MORE_FOLD per seed;
- 32 B_LESS_FOLD per seed;
- 384 anchors.

Reference:

- 64 hands × 8 boards;
- 512 deals/anchor.

Primary condition:

B_MORE_FOLD seed-cluster CI for `policy_value_b_minus_a_chips` must resolve negative.

If it resolves negative:
- inspect raw fold-vs-continue Advantage margins, fallback incidence, and target-estimator noise in this frozen structure;
- identify smallest intervention;
- keep holdout sealed until intervention is frozen.

If it remains unresolved:
- stop blind sample escalation;
- decompose residual variance.

If it resolves positive:
- do not train a fold fix;
- investigate hidden-chance/evaluation weighting.

No training before this gate.

Holdout `20261001..20261006` remains sealed.

## Immediate action

Run `bash tools/run_lt2_jammer_fai_structural_infoset_confirmation.sh`.

Stop at PASS or first error. Do not train.
