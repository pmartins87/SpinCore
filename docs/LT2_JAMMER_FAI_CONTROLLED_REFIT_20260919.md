# SpinCore — LT2 Jammer FAI controlled fresh-refit reservoir audit

Date: 2026-09-19  
Status: **ACTIVE — DISTINGUISH RESERVOIR SIGNAL DRIFT FROM LAST-FIT INSTABILITY**

## Trigger

The raw-margin decomposition resolved that, in B_MORE_FOLD:

- full B-A = `-24.549` chips;
- GAP_ONLY = `-18.719`, resolved;
- OFFSET_ONLY = `-4.946`, resolved;
- fallback-only argmax probe recovers `+15.415` versus B but does not establish Stage-A parity.

Therefore action-gap degradation is the larger causal component and fallback alone must not be patched.

## Question

Why is the Stage-B fold-vs-continue Advantage gap worse?

Two leading possibilities:

### H1 — reservoir / target-signal drift

The Stage-B HU Advantage reservoir contains a worse training signal for the fixed FAI structure.

Prediction:
- fresh models trained from Stage-B memory remain worse than fresh models trained from Stage-A memory under the same initialization and batch-sampling seeds, including at larger fit budgets.

### H2 — last-fit instability / insufficient fit

The Stage-B memory is not intrinsically worse; the particular final reset/refit produced a poor network.

Prediction:
- controlled fresh Stage-B refits recover most or all of the structural gap;
- larger budgets and/or replicate seeds reduce the B-vs-A difference.

## Frozen evaluation cohort

Reuse exactly the powered structural confirmation cohort:

- 384 anchors;
- 192 B_MORE_FOLD;
- 192 B_LESS_FOLD;
- legal `{FOLD,CALL}`;
- one public action before FAI.

Reconstruct the observations deterministically from the same forensic seeds and selection rule.

Reuse the already stored low-noise `q_canonical` reference from the structural JSON.

No low-noise reference recomputation is needed.

## Refit matrix

For each Stage-A and Stage-B HU Advantage reservoir:

- budgets: 100, 400, 1600 optimizer steps;
- 3 replicates each;
- same model initialization seed for A and B within each trial;
- same batch-sampling seed for A and B within each trial;
- vectorized batch path;
- canonical checkpoint learning rate and batch size.

Total in-memory optimizer work:

`2 × 3 × (100+400+1600) = 12,600` steps.

Source checkpoint files remain read-only.

## Primary outputs

For B_MORE_FOLD, per trial:

- Stage-B-reservoir minus Stage-A-reservoir policy value;
- canonical action-gap MSE difference;
- class-error-mass difference;
- fallback rates;
- policy regret;
- best-action agreement.

## Decision logic

If Stage-B reservoir remains consistently worse at 1600 steps:
- treat reservoir / target-signal drift as the leading cause;
- next inspect conditional training-target composition and weighting in this structural class.

If Stage-B reservoir catches Stage A with larger budgets:
- fit budget / optimizer convergence is implicated;
- design the smallest fit-schedule intervention.

If Stage-B results vary strongly across replicates while reservoir contrast is unstable:
- reset/fit seed variance is implicated;
- stabilize Advantage refitting before changing target generation.

If neither reservoir nor refit explains the production defect:
- inspect mismatch between sampled training distribution and this fixed evaluation cohort.

Holdout `20261001..20261006` remains sealed.
