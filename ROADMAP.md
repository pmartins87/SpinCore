# SpinCore Roadmap — active state 2026-09-19

## Active status

- LT0 — **DONE**.
- LT1 — **DONE**.
- LT2 Stage A — **PASS**.
- LT2 Stage B — **PASS** at 4.5M roots / iteration 7500.
- Weak-baseline regression — **CONFIRMED**.
- AveragePolicy extra-fit — **NOT SUPPORTED**.
- Advantage optimizer escalation — **NOT SUPPORTED GLOBALLY**.
- HU-preflop target variance — **HIDDEN/CHANCE DOMINANT**.
- K4 board averaging — **VALID ESTIMATOR IMPROVEMENT**.
- Jammer K4 causal gate — **NOT MET**.
- Cross-street future-chance audit — **PASS; FUTURE-CHANCE NOT FAILURE-SPECIFIC**.
- Stage-A/B target-drift matrix — **PASS; NO UNIVERSAL TARGET-DRIFT EXPLANATION**.
- HU policy-chain audit — **PASS; JAMMER DEFECT UPSTREAM IN CURRENT BEHAVIOR**.
- HU current-behavior first divergence — **PASS; 73.86% OF JAMMER LOSS AT PREFLOP FACING ALL-IN**.
- Broad Jammer FAI calibration — **PASS; SMALL SAMPLE INCONCLUSIVE**.
- Full-population Jammer FAI reconciliation — **PASS; EXPECTED FAI LOSS CARRIED BY B OVERFOLDING**.
- First fold-shift infoset audit — **PASS / UNDERPOWERED**.
- Powered structural infoset confirmation — **PASS; OVERFOLD CONFIRMED AT INFOSET LEVEL**.
- Raw center-vs-gap decomposition — **NEXT**.
- K4 training — **NOT AUTHORIZED**.
- long root training — **PAUSED**.
- DeepCrusher — **DEFERRED**.

## Preserved checkpoints

Stage A SHA:
`e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B SHA:
`3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Decisive structural infoset result

Frozen high-impact structure:

- Jammer FAI;
- legal `{FOLD,CALL}`;
- one public action before FAI.

B_MORE_FOLD policy-value B-A:

- `-24.54921`;
- CI95 `[-35.58425,-13.51416]`.

Fold shift:

- `+0.46907`;
- CI95 `[+0.45307,+0.48508]`.

Reference FOLD-minus-CONTINUE:

- `-40.86355`;
- CI95 `[-57.51783,-24.20927]`.

Canonical action-gap MSE B-A:

- `+0.000714714`;
- CI95 `[+0.000397781,+0.001031647]`.

Class-error mass B-A:

- `+0.127198`;
- CI95 `[+0.058510,+0.195887]`.

Fallback rate:

- A `6.25%`;
- B `50.00%`.

Raw-target MSE remains unresolved.

## Interpretation

The overfold is a real decision-time error against the hand-independent Jammer reference inside the previously localized structure.

The failure is not merely a sampled-action artifact, hidden-board artifact, or AveragePolicy-only artifact.

However, a fallback-only diagnosis is not yet sufficient because the canonical fold-vs-continue action gap also degrades significantly.

## Next gate — raw margin decomposition

Use the completed 384-anchor report only.

For legal raw outputs:

```
center = (fold + continue) / 2
gap    = fold - continue
```

Evaluate under exact production lean RM:

- B center + A gap = OFFSET_ONLY;
- A center + B gap = GAP_ONLY;
- B center + B gap = FULL_B.

Also measure a diagnostic-only argmax replacement for the all-nonpositive fallback.

Decision:

- GAP_ONLY dominant -> prioritize Advantage-gap fit / data diagnostics;
- OFFSET_ONLY dominant -> prioritize zero-crossing / fallback stabilization;
- both -> mixed cause;
- if fallback probe recovers only part of the loss, do not patch fallback alone.

Holdout `20261001..20261006` stays sealed.

## Immediate action

Run:

`bash tools/run_lt2_jammer_fai_raw_margin_decomposition.sh`.

Stop at PASS or first error. Do not train.
