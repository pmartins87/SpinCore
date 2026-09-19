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
- Fold-shift low-noise infoset audit — **NEXT**.
- K4 training — **NOT AUTHORIZED**.
- long root training — **PAUSED**.
- DeepCrusher — **DEFERRED**.

## Preserved checkpoints

Stage A SHA:
`e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B SHA:
`3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Full-population reconciliation verdict

Paired sampled FAI contribution:
- `-6.22672064777328`;
- exact reproduction of prior first-divergence result.

Deterministic expected FAI contribution:
- `-4.18434`;
- CI95 `[-6.58924,-1.77944]`.

Thus the FAI loss is real in expected policy value.

Conditional fold mass:
- A `0.24964`;
- B `0.37200`;
- B-A `+0.12237`;
- resolved.

By fold-shift direction:

- B_MORE_FOLD:
  - 7,374 rows;
  - contribution `-5.57116`;
  - CI95 `[-7.61160,-3.53071]`.

- B_LESS_FOLD:
  - 3,287 rows;
  - contribution `+1.38682`;
  - CI95 `[+0.21539,+2.55825]`.

- NO_FOLD_SHIFT:
  - contribution zero.

The damaging mechanism is specifically the Stage-B increase in FOLD probability.

## Structural localization

Legal `0,1`:
- contribution `-4.07764`, resolved.

Legal `0,1,9`:
- contribution `-0.11128`, unresolved.

Public path length 1:
- contribution `-3.52854`, resolved.

Path length 2:
- contribution `-0.65580`, unresolved.

Non-FOLD action values are equivalent after the Jammer all-in.

## Why one more diagnostic is required

The full-population reconciliation uses actual dealt hidden hand and full future board.

That is valid for population attribution but not for a decision-time infoset target.

Before changing training, verify that B_MORE_FOLD is also harmful under a low-noise infoset expectation.

## Next gate

Run `tools/run_lt2_jammer_fai_fold_shift_infoset.sh`.

Selection:
- classify common FAI states only by fold-mass shift sign;
- no sampled FAI action;
- no outcome;
- no realized Q;
- no low-noise Q used for selection.

Balanced sample:
- 8 B_MORE_FOLD per seed;
- 8 B_LESS_FOLD per seed;
- 96 anchors total.

Reference:
- 64 hands × 8 boards = 512 deals/anchor.

Primary metric:
`(sigma_B-sigma_A) dot Q_infoset`.

## Decision

If B_MORE_FOLD has resolved negative infoset policy value:
- overfold is a genuine decision-time model/policy error;
- inspect raw fold-vs-continue margins and fallback regimes inside that pre-registered group;
- design smallest isolated calibration intervention.

If B_MORE_FOLD is neutral/positive:
- do not train a fold fix;
- investigate hidden-chance/weighting covariance.

No training before this gate.

Holdout `20261001..20261006` remains sealed.

## Immediate action

Run `bash tools/run_lt2_jammer_fai_fold_shift_infoset.sh`.

Stop at PASS or first error. Do not train.
