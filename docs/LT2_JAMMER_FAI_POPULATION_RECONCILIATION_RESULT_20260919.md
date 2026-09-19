# SpinCore — LT2 Jammer FAI full-population reconciliation result

Date: 2026-09-19
Status: **PASS — FAI LOSS IS GENUINE IN EXPECTED POLICY VALUE; STAGE-B OVERFOLDING CARRIES THE LOSS**

## Integrity

Report schema:

`SPINCORE_LT2_JAMMER_FAI_POPULATION_RECONCILIATION_V1`.

The run used:

- all 27,170 HU Jammer seat-runs on forensic seeds `20260920..20260925`;
- 13,585 scenario clusters;
- no anchor subsampling;
- no training roots;
- no optimizer steps;
- no training-memory writes;
- holdout `20261001..20261006` untouched.

Status counts:

- COMMON_FAI: 16,279;
- EARLIER_HERO_DIVERGENCE: 8,840;
- NO_COMMON_FAI: 2,051.

Common-FAI reach rate:

`59.9153%`.

## Hard reproduction gate

The paired sampled FAI contribution reproduced the previous first-divergence result exactly:

- mean `-6.22672064777328` chips/hand;
- CI95 `[-9.25754,-3.19590]`;
- delta from prior result: `0.0`.

The reconciliation implementation therefore matches the prior paired stochastic attribution.

## Deterministic expected FAI contribution

Using every common FAI state and exact-deal terminal action values:

- mean `-4.18434` chips/hand;
- CI95 `[-6.58924,-1.77944]`.

This is resolved negative.

Therefore the FAI regression is **not** an artifact of sampled action noise.

It exists in deterministic expected Stage-B-minus-A policy value over the natural Jammer evaluation population.

Conditional on actually reaching a common FAI state:

- expected B-A `-6.9704` chips;
- seed-cluster CI95 `[-11.2658,-2.6751]`.

## Fold-mass shift

Conditional on common FAI:

- Stage A mean fold mass: `0.24964`;
- Stage B mean fold mass: `0.37200`;
- B-A: `+0.12237`;
- seed-cluster CI95 `[+0.11779,+0.12695]`.

Stage B folds roughly **12.24 percentage points more**.

## Causal decomposition by fold-shift direction

### B_MORE_FOLD

- 7,374 common FAI rows;
- additive expected contribution:
  `-5.57116` chips/hand;
- CI95 `[-7.61160,-3.53071]`.

This is the dominant harmful component.

### B_LESS_FOLD

- 3,287 common FAI rows;
- additive expected contribution:
  `+1.38682`;
- CI95 `[+0.21539,+2.55825]`.

When Stage B folds less, the shift is beneficial on average.

### NO_FOLD_SHIFT

- 5,618 rows;
- contribution numerically zero.

Thus the resolved FAI loss is specifically carried by **Stage B increasing fold probability**.

## Structural decomposition

### Legal actions {FOLD, CALL}

7,892 common FAI rows.

Contribution:
- `-4.07764`;
- CI95 `[-5.88461,-2.27067]`.

This block alone explains almost all of the deterministic expected loss.

### Legal actions {FOLD, CALL, ALL_IN}

8,356 rows.

Contribution:
- `-0.11128`;
- CI95 `[-1.76142,+1.53886]`.

Unresolved and near neutral.

### Rare {0,1,5,9}

31 rows.

Near zero.

## Path length

One public action before FAI:
- 13,579 rows;
- contribution `-3.52854`;
- CI95 `[-5.83127,-1.22581]`.

Two public actions:
- 2,700 rows;
- contribution `-0.65580`;
- CI crosses zero.

The principal loss is therefore concentrated in the simplest FAI structure.

## Outcome-equivalence confirmation

Maximum non-FOLD actual-Q spread:

`0.0`.

After the opponent is already all-in, CALL and ALL_IN are terminal-value equivalent in this population.

Therefore the strategically relevant class is FOLD versus CONTINUE, not CALL versus ALL_IN slot identity.

## Reconciliation with the 48-anchor broad audit

The earlier 48-anchor low-noise audit was not wrong; it was too small / heterogeneous to reveal the full-population failure reliably.

The full population resolves:

- genuine negative expected FAI value;
- a +12.24 pp fold shift;
- harm specifically when Stage B folds more.

The next question is no longer whether FAI matters.

The next question is:

**Does Stage B overfold the same states under a low-noise infoset reference, or is the full-population actual-deal loss driven by realized hidden-card/board variance?**

## Next gate

Run the pre-registered fold-shift low-noise infoset audit:

`tools/run_lt2_jammer_fai_fold_shift_infoset.sh`.

Selection is based only on production fold-mass shift sign:

- B_MORE_FOLD;
- B_LESS_FOLD.

It does not use:
- sampled FAI action;
- terminal outcome;
- actual-deal Q;
- low-noise reference.

Per seed:
- 8 B_MORE_FOLD anchors;
- 8 B_LESS_FOLD anchors.

Total:
- 96 anchors.

Reference:
- 64 uniform compatible opponent hands;
- 8 future boards/hand;
- 512 deals/anchor.

Primary quantity:

`policy_value_B_minus_A = (sigma_B-sigma_A) dot Q_infoset`.

No training is authorized.
