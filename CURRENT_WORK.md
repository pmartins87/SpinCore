# SpinCore Current Work

Date: 2026-09-19
Status: **LT2 STAGE B PASS — BROAD JAMMER FAI CALIBRATION DOES NOT SHOW STAGE-B POLICY-REGRET DEGRADATION — FIRST-DIVERGENCE CONTRADICTION MUST BE RECONCILED ON FULL POPULATION — NO TRAINING**

## Active source of truth

Read before new compute:

- `docs/LT2_JAMMER_FAI_BROAD_CALIBRATION_RESULT_20260919.md`
- `docs/LT2_JAMMER_FAI_POPULATION_RECONCILIATION_20260919.md`
- `docs/LT2_HU_BEHAVIOR_FIRST_DIVERGENCE_RESULT_20260919.md`
- `docs/LT2_HU_POLICY_CHAIN_RESULT_20260918.md`
- `docs/LONG_TRAINING_PLAN.md`

Preserve Stage A and Stage B. Do not continue root training beyond iteration 7500.

## Preserved checkpoints

Stage A:
- iteration 3000 / 1.8M roots;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Broad Jammer FAI calibration result

48 broad common FAI anchors, 8 per forensic seed, selected before the FAI action and without conditioning on FAI A/B divergence or terminal outcome.

### Policy regret

Stage A:
- `32.885` chips;
- seed-cluster CI95 `[22.073,43.697]`.

Stage B:
- `20.984`;
- seed-cluster CI95 `[11.894,30.073]`.

B-A:
- `-11.901`;
- seed-cluster CI95 `[-25.515,+1.713]`.

Stage B is not broadly worse in policy regret. Direction is better, but unresolved.

### Canonical action-gap MSE

Stage A:
- `0.0020185`.

Stage B:
- `0.0014322`.

B-A:
- `-0.00058635`;
- CI95 `[-0.00093271,-0.00023998]`.

Resolved Stage-B improvement.

### FOLD-vs-CONTINUE class-error mass

Stage A:
- `0.38346`.

Stage B:
- `0.29914`.

B-A:
- `-0.08432`;
- CI95 `[-0.16662,-0.00203]`.

Resolved Stage-B improvement.

### Fold mass

Stage A:
- `0.27466`.

Stage B:
- `0.37475`.

B-A:
- `+0.10009`;
- CI95 `[+0.04693,+0.15326]`.

Stage B folds about 10 pp more on the broad sample, but the class-error metric improves, so the fold shift alone is not evidence of a defect.

### Fallback

All-nonpositive fallback:
- Stage A `5/48 = 10.42%`;
- Stage B `12/48 = 25.00%`.

Fallback is more frequent in B, but this does not align with worse class error or action-gap MSE.

## Positive-support metric correction

The strict positive-support/Jaccard metric is not valid as standalone causal evidence.

Reason:

`A*_S(a)=Q(a)-V_{sigma_S}`.

When `sigma_S` is pure on an optimal action, that selected action has true Advantage exactly zero, while the raw model generally needs a positive value to induce the pure regret-matching action.

Thus `raw>0` versus `A*>0` can label a policy-consistent optimal action as a false positive.

Do not use zero exact-support rate or Stage-B Jaccard decline as a training target.

## Scientific contradiction

Previous full paired first divergence:

- Jammer current behavior B-A `-8.4304`, resolved;
- FAI additive contribution `-6.2267`, resolved.

Broad 48-anchor calibration:

- Stage B policy regret not worse;
- canonical gap MSE improves;
- class-error mass improves.

Possible causes:
1. 48 anchors underpowered / unrepresentative;
2. loss concentrated in a high-leverage subset;
3. sampled first-divergence attribution needs direct deterministic policy-value reconciliation.

Before any intervention, reconcile them on every natural evaluation Jammer FAI state.

## Active gate

Run:

```bash
bash tools/run_lt2_jammer_fai_population_reconciliation.sh
```

This uses every HU Jammer seat-run on the forensic seeds.

At each common FAI state:
- no anchor subsampling;
- exact dealt hidden hand/full board retained;
- every legal action applied to a cloned solver state;
- terminal chip delta read directly;
- deterministic expected policy-value delta
  `sum (sigma_B-sigma_A) Q_actual`;
- paired sampled action contribution reproduced using the exact prior RNG.

The sampled contribution must exactly reproduce:
`-6.22672064777328`.

Holdout `20261001..20261006` remains untouched.

DeepCrusher remains deferred.

## Immediate user action

Pull `main` and run:

```bash
bash tools/run_lt2_jammer_fai_population_reconciliation.sh
```

Wait for `LT2_JAMMER_FAI_POPULATION_RECONCILIATION_PASS` or the first error.

Then send `SpinCore_LT2_jammer_fai_population_reconciliation.json`.

Do not start any training.
