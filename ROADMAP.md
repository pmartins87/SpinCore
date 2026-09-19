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
- Broad Jammer FAI calibration — **PASS; DOES NOT SHOW BROAD STAGE-B POLICY-REGRET DEGRADATION**.
- Full-population Jammer FAI counterfactual reconciliation — **NEXT**.
- K4 training — **NOT AUTHORIZED**.
- long root training — **PAUSED**.
- DeepCrusher — **DEFERRED**.

## Preserved checkpoints

Stage A SHA:
`e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B SHA:
`3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Broad FAI verdict

The 48-anchor broad low-noise calibration does not support:

`Stage B broadly worse at FAI -> Jammer loss`.

Policy regret:
- A `32.885`;
- B `20.984`;
- B-A `-11.901`, CI crosses zero.

Canonical action-gap MSE:
- B-A `-0.00058635`, resolved improvement.

FOLD-vs-CONTINUE class-error mass:
- B-A `-0.08432`, resolved improvement.

Fold mass:
- B-A `+0.10009`, resolved.

Fallback incidence:
- A 10.42%;
- B 25.00%.

Fallback frequency rises, but Stage-B class error and gap MSE improve. It cannot be called the cause.

Strict positive-support Jaccard is methodologically unsuitable because stage-specific true Advantage can be exactly zero on a pure optimal action while the raw model must be positive there to induce that action.

## Remaining contradiction

Full paired first divergence says:
- FAI contributes `-6.2267` chips/hand to Jammer Stage-B-minus-A current behavior.

Broad 48-anchor expected regret says:
- Stage B is not worse.

Resolve this before changing training.

## Next gate

Run `tools/run_lt2_jammer_fai_population_reconciliation.sh`.

Use every natural HU Jammer seat-run.

For every common FAI state:
- compute actual-deal terminal Q for every legal action;
- compute deterministic `(sigma_B-sigma_A) dot Q_actual`;
- also reproduce the prior paired sampled contribution with exact RNG.

Hard validation:
- sampled additive FAI contribution must equal `-6.22672064777328`.

## Decision

If deterministic full-population expected FAI contribution is resolved negative:
- first-divergence loss is genuine in expected policy value;
- identify which fold-mass shift regime carries it;
- then apply low-noise infoset references to a pre-registered subset.

If deterministic expected contribution is neutral/positive:
- correct the previous sampled attribution before any intervention.

No training before reconciliation.

Holdout `20261001..20261006` remains sealed.

## Immediate action

Run `bash tools/run_lt2_jammer_fai_population_reconciliation.sh`.

Stop at PASS or first error. Do not train.
