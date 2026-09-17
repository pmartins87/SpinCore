# SpinCore — LT2 30k weak-baseline variance result

Date: 2026-09-17
Status: **PRECISION TARGET MET — HU JAMMER CONFIRMED NEGATIVE — TRAINING-DYNAMICS AUDIT NEXT**

## Design

Six independent seed blocks (`20260920` through `20260925`), 5000 scenarios each, for 30,000 scenarios per checkpoint. Stage A and Stage B used identical scenario/deal/opponent/hero RNG streams within each seed. No training occurred.

The six primary Stage-B claims use Bonferroni simultaneous family-wise 95% confidence intervals. The predeclared precision target was a worst primary simultaneous half-width <= 5 chips/hand; observed maximum was **4.963**, so the target was met.

## Stage B primary result

| Opponent | Domain | Stage B chips/hand | simultaneous family-wise 95% CI | Classification |
|---|---|---:|---:|---|
| UNIFORM_LEGAL | 3H | +20.102 | [+16.372,+23.831] | POSITIVE |
| UNIFORM_LEGAL | HU | +21.484 | [+16.521,+26.447] | POSITIVE |
| PASSIVE_CALLER | 3H | +4.030 | [+0.977,+7.082] | POSITIVE |
| PASSIVE_CALLER | HU | -0.996 | [-4.435,+2.442] | UNRESOLVED |
| JAMMER | 3H | +0.859 | [-2.628,+4.347] | UNRESOLVED |
| JAMMER | HU | **-5.141** | **[-9.078,-1.204]** | **NEGATIVE** |

The apparent HU Jammer weakness from the small pilot was therefore **not only variance**. With the predeclared precision design, Stage B has a statistically resolved negative raw chip EV against the HU Jammer family.

The passive-HU and jammer-3H cells are near zero at the achieved precision and remain unresolved; they must not be relabeled positive or negative without additional evidence.

## Stage A -> Stage B paired movement

The 30k common-random design also permits a much more sensitive Stage-B-minus-Stage-A comparison.

Ordinary paired 95% intervals show Stage B moved downward against several weak-opponent cells:

- Jammer HU: `-1.682` chips/hand, 95% CI `[-2.767,-0.597]`;
- Jammer 3H: `-0.923`, 95% CI `[-1.729,-0.116]`;
- Passive HU: `-1.261`, 95% CI `[-2.377,-0.145]`.

Because these are six related A/B comparisons, the family-wise interpretation must also control multiplicity. Applying the same six-claim Bonferroni z (`2.638257`) to the paired standard errors gives:

- Jammer HU: **[-3.143,-0.222]**, still negative after family-wise control;
- Jammer 3H: `[-2.008,+0.163]`, unresolved after family-wise control;
- Passive HU: `[-2.763,+0.241]`, unresolved after family-wise control;
- the remaining three cells also include zero.

Thus the strongest checkpoint-learning conclusion is specific: **the extra Stage-A -> Stage-B training materially worsened HU performance against Jammer under this weak-baseline diagnostic**, and this conclusion survives simultaneous family-wise control. This does not imply that every aspect of Stage B is worse; it identifies a real failure mode that further blind root count should not be expected to fix automatically.

## Interpretation

The current evidence now separates variance from training behavior:

- uniform-legal performance is clearly positive in both domains;
- passive-caller 3H is clearly positive;
- passive-caller HU and jammer 3H are approximately near-zero at current precision;
- jammer HU is genuinely negative;
- Stage B changed materially from Stage A, but the change did not uniformly improve weak-opponent strength and specifically degraded the HU-vs-Jammer cell.

Therefore the next question is not “should we add more roots?” and not “should we benchmark DeepCrusher?” The next question is **why the training approximation produced this weak-opponent behavior, especially in HU**.

## Immediate diagnostic

Run the read-only checkpoint fit audit:

`tools/run_lt2_training_dynamics_fit_audit.sh`

It samples the stored Advantage and AveragePolicy memories from both Stage A and Stage B and reports, separately for 3H/HU:

- reservoir size/seen counts and sample iteration-age distribution;
- Advantage legal-action MSE versus a zero predictor;
- Advantage-induced regret-matching policy TV and argmax agreement;
- AveragePolicy target entropy, model cross-entropy and excess KL;
- AveragePolicy TV and argmax agreement;
- optimizer/reset counters.

There is deliberately **no arbitrary PASS threshold** in this audit. The observed fit quality will determine whether the next bounded experiment should increase Advantage fit budget, AveragePolicy fit budget, or look elsewhere in the learning semantics.

## Stop condition

Do not resume long training beyond iteration 7500 until the fit audit is reviewed. DeepCrusher remains deferred.