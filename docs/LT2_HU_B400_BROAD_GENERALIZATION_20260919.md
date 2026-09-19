# SpinCore — LT2 HU B400 broad generalization gate

Date: 2026-09-19  
Status: **ACTIVE — TEST MINIMUM SUFFICIENT 400-STEP REFIT ON NATURAL HU FORENSIC POPULATION**

## Trigger

Controlled fresh refits on the fixed Jammer FAI cohort showed:

- 100 steps: Stage-B reservoir remains worse than Stage-A reservoir;
- 400 steps: all three paired Stage-B refits become better;
- 1600 steps: all three remain better.

The Stage-B reservoir therefore contains usable signal; canonical 100-step fitting is the leading failure mechanism.

## Why 400, not 1600

400 is the smallest tested budget that passed the structural cohort in every replicate.

It costs approximately 4x the canonical optimizer work rather than 16x.

The next question is whether that smaller intervention generalizes beyond the selected FAI cohort.

## Candidate construction

Recreate the three exact Stage-B 400-step refits from the controlled-refit report.

For each candidate:

- source reservoir: preserved Stage B at iteration 7500;
- model reset initialization seed: same as the corresponding controlled trial;
- batch RNG seed: same as the corresponding controlled trial;
- 400 steps;
- batch size 1024;
- learning rate 0.001;
- vectorized batches.

No candidate seed is selected by outcome.

## Broad evaluation population

Use all HU scenarios generated from forensic seeds `20260920..20260925` at 5000 scenarios/seed.

Against:

- UNIFORM_LEGAL;
- PASSIVE_CALLER;
- JAMMER.

Policies compared with common random numbers:

- PROD_A current Advantage behavior;
- PROD_B current Advantage behavior;
- B400_R0;
- B400_R1;
- B400_R2.

This is a natural-population evaluation, not the structurally selected FAI cohort.

## Primary gate

For JAMMER:

- each B400 replicate must improve versus PROD_B in paired chip EV;
- prefer CI95 entirely above zero for each replicate;
- inspect B400 vs PROD_A to determine whether the Stage-B current-behavior regression has been removed or merely reduced.

## Secondary safety checks

For UNIFORM_LEGAL and PASSIVE_CALLER:

- no B400 replicate should show a clearly resolved material deterioration versus PROD_B.

If one weak baseline regresses while Jammer improves, do not automatically reject; quantify the tradeoff and inspect whether the effect is statistically resolved and strategically coherent.

## Decision branches

### B400 passes broadly

Freeze `advantage_steps=400` as the candidate training intervention.

Next:
- run a bounded continuation pilot from Stage B;
- do not yet open holdout;
- compare learning dynamics and global weak-baseline behavior after the pilot.

### B400 fixes FAI but fails broad Jammer

Escalate one time to the already-supported 1600-step candidate and repeat the broad gate.

### B400 improves Jammer but materially harms another baseline

Do not resume long training.
Investigate whether fit budget is exposing a multi-objective conflict or whether AveragePolicy buffering is needed.

### B400 is unstable across its three deterministic replicates

Treat reset/fit variance as still material and stabilize fitting before root continuation.

Holdout `20261001..20261006` remains sealed.
