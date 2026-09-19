# SpinCore Roadmap — active state 2026-09-19

## Active status

- LT0 — **DONE**.
- LT1 — **DONE**.
- LT2 Stage A — **PASS**.
- LT2 Stage B — **PASS** at 4.5M roots / iteration 7500.
- Weak-baseline regression — **CONFIRMED**.
- HU policy-chain — **JAMMER DEFECT UPSTREAM IN CURRENT ADVANTAGE BEHAVIOR**.
- Jammer FAI first divergence — **73.86% OF CURRENT-BEHAVIOR LOSS AT PREFLOP FAI**.
- Full-population FAI reconciliation — **OVERFOLD CARRIES RESOLVED LOSS**.
- First low-noise fold-shift audit — **UNDERPOWERED**.
- Powered structural infoset confirmation — **PASS; OVERFOLD CONFIRMED AT INFOSET LEVEL**.
- Raw center-vs-gap decomposition — **PASS; ACTION-GAP DRIFT DOMINANT, OFFSET/FALLBACK SECONDARY**.
- Fallback-only patch — **NOT AUTHORIZED**.
- Controlled Stage-A/B reservoir refit audit — **NEXT**.
- K4 training — **NOT AUTHORIZED**.
- long root training — **PAUSED**.
- DeepCrusher — **DEFERRED**.

## Preserved checkpoints

Stage A SHA:
`e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B SHA:
`3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Resolved causal chain

1. Deployed AveragePolicy worsens vs Jammer.
2. A much larger Jammer defect exists in current Advantage-induced behavior.
3. 73.86% of that current-behavior loss first appears at preflop FAI.
4. Full population shows Stage B folds more and loses expected value there.
5. Powered low-noise infoset confirmation shows this is a genuine decision-time overfold, not hidden-card/board covariance.
6. The fixed structural cohort resolves worse canonical action-gap fit and class error at Stage B.
7. Raw decomposition shows:
   - GAP_ONLY `-18.719`, resolved;
   - OFFSET_ONLY `-4.946`, resolved;
   - fallback-only argmax recovers part but not all of the loss.

The leading unresolved question is now upstream of policy mapping:

**did the Stage-B Advantage reservoir become worse for this structural class, or did the final reset/refit fail to extract the available signal?**

## Next gate

Controlled fresh refit on the same fixed 384 anchors.

For Stage-A and Stage-B HU Advantage reservoirs:

- 100 / 400 / 1600 optimizer steps;
- 3 paired replicates;
- same init seed A/B within each trial;
- same batch-sampling seed A/B within each trial;
- vectorized batches;
- canonical learning rate and batch size.

No roots.

Interpretation:

- B reservoir still worse at 1600 -> reservoir/target-signal drift;
- B catches A with more fit -> optimizer budget/convergence;
- strong replicate instability -> reset/fit seed variance;
- none -> training-distribution mismatch.

Holdout `20261001..20261006` remains sealed.

## Immediate action

Run `bash tools/run_lt2_jammer_fai_controlled_refit.sh`.

Do not train beyond this in-memory audit.
