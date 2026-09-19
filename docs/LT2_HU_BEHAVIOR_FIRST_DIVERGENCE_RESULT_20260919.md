# SpinCore — LT2 HU current-behavior first-divergence result

Date: 2026-09-19
Status: **PASS — JAMMER CURRENT-BEHAVIOR LOSS LOCALIZES PRIMARILY TO PREFLOP FACING ALL-IN; BROAD NON-SELECTED ACTION-GAP / REGRET-MATCHING CALIBRATION NEXT**

## Integrity

Report schema:

`SPINCORE_LT2_HU_BEHAVIOR_FIRST_DIVERGENCE_V1`.

The audit used:

- forensic seeds `20260920..20260925`;
- 5000 scenarios/seed;
- 13,585 HU scenario clusters;
- 27,170 Jammer seat-runs;
- identical scenario, deal, hero seat, baseline and RNG streams until first Stage-A/B current-behavior divergence;
- no training roots;
- no optimizer steps;
- no training-memory writes;
- holdout `20261001..20261006` untouched.

## JAMMER total current-behavior regression

Stage-B minus Stage-A current behavior:

- mean `-8.4304` chips/hand;
- 95% CI `[-12.5155,-4.3453]`.

Divergence rate:

- `59.78%`.

## First-divergence decomposition

### PREFLOP_FACING_ALL_IN

- seat-run frequency: `27.24%`;
- 7402 seat-runs;
- contribution `-6.2267` chips/hand;
- 95% CI `[-9.2575,-3.1959]`.

This single group explains:

`-6.2267 / -8.4304 = 73.86%`

of the total resolved Jammer current-behavior regression.

### PREFLOP_ROOT

- seat-run frequency: `32.54%`;
- 8840 seat-runs;
- contribution `-2.2037`;
- 95% CI `[-4.9947,+0.5873]`.

This accounts numerically for the remaining `26.14%`, but is not individually resolved.

### Postflop

For Jammer:

- FLOP: 0 seat-runs;
- TURN: 0;
- RIVER: 0;
- PREFLOP_OTHER: 0.

Thus the entire paired Jammer current-behavior difference is preflop.

## Transition structure inside FACING_ALL_IN

Clearly outcome-changing FOLD/non-FOLD transitions include:

- `0->1`: 772;
- `0->9`: 590;
- `1->0`: 3335;
- `9->0`: 64.

Total clearly FOLD-vs-CONTINUE:

- 4761 / 7402 = **64.32%**.

Raw slot-equivalent or potentially outcome-equivalent transitions include:

- `1->9`: 2259;
- `9->1`: 367;
- rare transitions involving slot 5: 15 total.

Therefore raw slot divergence must not be equated with strategic-value divergence. The next audit uses expected value under a common low-noise action-value reference rather than sampled slot mismatch alone.

## Causal synthesis

The following evidence now aligns:

1. global policy-chain audit:
   - Jammer current behavior B-A `-8.430`, resolved;

2. current-behavior first divergence:
   - `73.86%` of that loss localizes to `PREFLOP_FACING_ALL_IN`;

3. prior target-drift matrix:
   - Jammer FAI conditional target is effectively invariant Stage A -> Stage B;

4. prior own-target fit:
   - Stage-B global FAI target MSE did not show a matching deterioration;

5. K4 experiments:
   - board averaging improves target estimation, but estimator noise is not failure-specific.

Therefore the highest-value hypothesis is now:

**small / structured Advantage action-gap or sign errors are being amplified by production regret matching at Jammer FAI states, even though aggregate MSE does not worsen.**

## Next gate

Run a broad, non-divergence-selected Jammer FAI calibration.

Selection:

- all paired Stage-A/B current-behavior trajectories that reach a common Jammer FAI state before any earlier A/B sampled-action divergence;
- record the state **before** sampling the FAI hero action;
- deterministic balanced sample by seed;
- selection does not depend on whether A/B later diverge at FAI or on terminal outcome.

For every anchor build a common low-noise action-value reference using:

- uniform compatible opponent hands, because Jammer is hand-independent;
- uniform future boards;
- canonical `Q(a)-mean_legal(Q)` gauge.

For each Stage A/B raw Advantage model:

- derive exact production regret-matching policy;
- derive stage-specific true Advantage target
  `Q(a)-sum sigma(b)Q(b)`;
- raw target MSE;
- canonical action-gap MSE;
- positive-support/sign mistakes;
- all-nonpositive softmax fallback incidence;
- best-action agreement;
- policy mass on truly negative actions;
- expected policy regret.

This directly tests the nonlinear RM amplification hypothesis without selecting failure states.

Canonical launcher:

`bash tools/run_lt2_jammer_fai_broad_calibration.sh`.

No training is authorized.
