# SpinCore — LT2 Jammer FAI controlled fresh-refit result

Date: 2026-09-19  
Status: **PASS — STAGE-B RESERVOIR IS NOT THE FAILURE; CANONICAL 100-STEP REFIT IS INSUFFICIENT FOR THE MATURE RESERVOIR**

## Integrity

Report schema:

`SPINCORE_LT2_JAMMER_FAI_CONTROLLED_REFIT_V1`.

The audit used:

- the preserved Stage-A HU Advantage reservoir;
- the preserved Stage-B HU Advantage reservoir;
- the same fixed 384-anchor low-noise structural cohort;
- identical initialization seeds across A/B within each trial;
- identical batch-sampling seeds across A/B within each trial;
- budgets 100, 400 and 1600 optimizer steps;
- 3 replicates per budget;
- vectorized batches, batch size 1024, learning rate 0.001;
- no new training roots;
- no source-checkpoint mutation;
- no source training-memory writes;
- holdout `20261001..20261006` untouched.

Stage A reservoir:
- 2,000,000 stored items;
- 24,655,892 samples seen.

Stage B reservoir:
- 2,000,000 stored items;
- 62,622,782 samples seen.

## Production reproduction

The fixed B_MORE_FOLD cohort reproduced the production defect:

- policy value B-A `-24.54921` chips;
- CI95 `[-35.58425,-13.51416]`;
- gap-MSE B-A `+0.000714714`, resolved;
- Stage-A fallback `6.25%`;
- Stage-B fallback `50.00%`.

The evaluation cohort and reference therefore remained aligned with the prior structural gate.

## 100-step refits — canonical budget still fails

Across the three paired fresh-refit replicates:

Stage-B-reservoir minus Stage-A-reservoir policy value on B_MORE_FOLD:

- mean `-11.64417` chips;
- replicate-mean CI95 `[-18.26748,-5.02086]`.

Individual trials:

- rep 0: `-16.8111`;
- rep 1: `-12.8336`;
- rep 2: `-5.2878`.

Thus the failure is reproduced by fresh 100-step fitting. It is not peculiar to the one production reset seed at iteration 7500.

## 400-step refits — sign reverses robustly

Across the three paired replicates:

- Stage-B-reservoir minus Stage-A-reservoir policy value:
  `+13.91919` chips;
- replicate-mean CI95:
  `[+2.96579,+24.87259]`.

Individual trials:

- rep 0: `+24.9640`;
- rep 1: `+9.8810`;
- rep 2: `+6.9126`.

All three 400-step Stage-B refits outperform the corresponding Stage-A-reservoir refit on the primary structural cohort.

The mean Stage-B-vs-A gap-MSE difference is approximately neutral / slightly favorable to B and unresolved.

## 1600-step refits — stable high-fit regime

Across the three paired replicates:

- Stage-B-reservoir minus Stage-A-reservoir policy value:
  `+7.29093`;
- replicate-mean CI95:
  `[+3.38533,+11.19653]`.

Individual trials:

- rep 0: `+3.5550`;
- rep 1: `+7.9571`;
- rep 2: `+10.3607`.

At 1600 steps the Stage-B fresh models also show much lower absolute regret than the corresponding Stage-A fresh models on this cohort.

## Verdict

The Stage-B reservoir is **not intrinsically worse** for the confirmed Jammer FAI structural class.

The decisive pattern is:

- 100 steps: B remains worse;
- 400 steps: B becomes better in every replicate;
- 1600 steps: B remains better in every replicate.

This strongly implicates **insufficient Advantage refit budget / convergence at the mature 2M-item Stage-B reservoir**.

It also rejects the simpler hypothesis that more Stage-B roots poisoned the reservoir signal.

The failure is not merely one unlucky production initialization: all three fresh 100-step replicates remain negative.

## Budget choice for the next gate

Do not jump directly to 1600 steps in long training.

400 is the smallest tested budget that clears the structural defect in all three replicates and is approximately 4x the optimizer work instead of 16x.

Therefore 400 is the candidate intervention for a broad forensic generalization gate.

1600 remains the escalation branch only if 400 fails to generalize.

## Next gate

Generate the same three deterministic 400-step Stage-B refits from the preserved Stage-B reservoir and evaluate them on the complete forensic HU population against:

- UNIFORM_LEGAL;
- PASSIVE_CALLER;
- JAMMER.

Pair every candidate with production Stage A and production Stage B on identical scenario, deal, seat and RNG streams.

Acceptance direction:

- all three B400 candidates should improve the global Jammer result versus production Stage B;
- they must not show a resolved material regression against the other weak baselines;
- Stage-A comparison is contextual, not a hard requirement unless the Jammer result remains clearly below Stage A.

No holdout seeds are opened.
