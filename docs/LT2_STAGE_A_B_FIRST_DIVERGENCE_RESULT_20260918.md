# SpinCore — LT2 Stage-A -> Stage-B first-divergence forensic result

Date: 2026-09-18
Status: **COMPLETE — HU-JAMMER REGRESSION LOCALIZED MAINLY TO PREFLOP FACING ALL-IN; REGRESSION IS NOT UNIVERSALLY PREFLOP**

## Source and integrity

Read-only paired forensic using the already-seen diagnostic seed family:

- 20260920
- 20260921
- 20260922
- 20260923
- 20260924
- 20260925

Default 5,000 full-sampler scenarios per seed.

Observed:
- 13,585 HU scenarios;
- 81,510 seat-runs across three weak baselines;
- Stage A and Stage B source checkpoints unchanged;
- no training roots;
- no optimizer steps.

The lightweight-policy resource fix worked: each worker loaded only ~0.59 MiB HU AveragePolicy snapshots.

## Overall paired Stage-B minus Stage-A

### UNIFORM_LEGAL

- B-A: `-0.416` chips/hand;
- 95% CI `[-1.812,+0.979]`;
- divergence rate: `9.3%`.

Overall regression is unresolved.

A resolved negative subgroup nevertheless appears at TURN:
- contribution `-0.552`;
- 95% CI `[-0.970,-0.134]`.

Other groups partially offset it.

### PASSIVE_CALLER

- B-A: `-1.261`;
- 95% CI `[-2.377,-0.145]`;
- divergence rate: `27.8%`.

Largest resolved contribution:
- FLOP: `-0.672`;
- 95% CI `[-1.299,-0.046]`.

This is about **53%** of the total PassiveCaller regression.

The rest is distributed across root/turn/river with individual intervals crossing zero.

### JAMMER

- B-A: `-1.682`;
- 95% CI `[-2.767,-0.597]`;
- divergence rate: only `4.8%`.

The entire observed regression localizes to preflop:

- PREFLOP_ROOT:
  - frequency `2.3%`;
  - contribution `-0.557`;
  - 95% CI `[-1.139,+0.026]` — close but not individually resolved.

- PREFLOP_FACING_ALL_IN:
  - frequency `2.5%`;
  - contribution `-1.126`;
  - 95% CI `[-2.049,-0.202]` — **resolved negative**.

The FACING_ALL_IN component alone explains about **67%** of the total HU-Jammer A->B regression.

There are no postflop first-divergence contributions for Jammer in this audit.

## Main interpretation

This is the first direct evidence that the confirmed HU-Jammer regression manifests in exactly the class of states where the independent target diagnostics had already found a strong variance problem:

**HU preflop after the opponent jams.**

That makes the K4 board-averaging hypothesis substantially more plausible as a causal contributor to the Jammer regression.

However, this does **not** prove K4 is the global Stage-B fix.

The forensic simultaneously shows a separate resolved PassiveCaller FLOP regression and a resolved UNIFORM_LEGAL TURN subgroup. Therefore the Stage-A -> Stage-B deterioration is not one universal HU-preflop phenomenon.

A K4 HU-preflop intervention can at most address the Jammer-facing preflop mechanism unless later evidence shows broader indirect effects.

## Anti-overfitting implication

The overlap was not created by tuning directly to the Jammer benchmark:

1. hidden/future-board target variance was diagnosed independently;
2. K4 was selected from compute-normalized target-estimator experiments;
3. the deployed-policy forensic was then run independently and localized ~67% of Jammer regression to FACING_ALL_IN.

This triangulation is stronger than benchmark-specific tuning.

Still, one causal link remains missing:

`noisy Advantage targets -> wrong current Advantage behavior -> wrong AveragePolicy shift -> negative Jammer EV`.

The next audit must measure that chain on the **actual forensic FACING_ALL_IN divergence states**.

## Decision

Do not train K4 yet.

Run a targeted matched-state Advantage/target overlay on actual Jammer FACING_ALL_IN first-divergence states.

The overlay must compare Stage A and Stage B, on identical observable states:

- stored AveragePolicy action probabilities;
- current Advantage raw outputs and induced regret-matching policy;
- high-budget information-set conditional target reference;
- model/reference TV;
- sign/branch mismatch;
- FOLD / CHECK_CALL / ALL_IN mass;
- target-policy value gap/regret;
- K1 versus K4 board-only estimator error on those exact states.

Only if Stage B is demonstrably farther from the low-noise target in the same direction as its deployed AveragePolicy regression, and K4 reduces the label error there, may a bounded K4 training pilot be considered.

Even then, long training must remain paused until the separate postflop regression is accounted for.
