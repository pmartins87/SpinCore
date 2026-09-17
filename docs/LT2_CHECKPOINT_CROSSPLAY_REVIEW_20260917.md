# SpinCore — LT2 Stage A -> Stage B checkpoint cross-play review

Date: 2026-09-17
Status: **INITIAL CROSS-PLAY BORDERLINE / MIXED — HIGHER-POWER FRESH-SEED CONFIRMATION REQUIRED**

## Inputs

Stage A:

- iteration 3000 / 1.8M roots;
- checkpoint SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B:

- iteration 7500 / 4.5M roots;
- checkpoint SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Evaluation:

- 3000 fresh empirical scenarios;
- seed `20260918`;
- 1640 3H / 1360 HU scenarios;
- 31 workers;
- same scenario, deal, hero seat and per-seat RNG streams for paired comparisons;
- read-only, no training.

## Primary mixture result

Primary metric is Stage B hero minus Stage A hero against the exact same deterministic 50/50 Stage-A/Stage-B opponent mixture.

| Domain | B - A chips/hand | 95% CI |
|---|---:|---:|
| ALL | -1.8129 | [-3.9564, +0.3306] |
| 3H | -1.9760 | [-4.3207, +0.3687] |
| HU | -1.6162 | [-5.4070, +2.1747] |

All three point estimates favor Stage A, but all three 95% CIs still include zero. ALL and 3H are borderline: their upper bounds are only about +0.33/+0.37 chips per hand.

## Additional diagnostics

HU direct seat-balanced Stage B vs Stage A:

- Stage B chip EV `-0.6165`;
- 95% CI `[-10.7519, +9.5188]`.

3H invasion:

- B singleton vs A/A: `+1.9612` chips/hand, CI `[-5.6098,+9.5322]`;
- A singleton vs B/B: `-0.9283` chips/hand, CI `[-8.5324,+6.6759]`;
- invasion difference B-vs-AA minus A-vs-BB: `+2.8894`, CI `[-1.2160,+6.9949]`.

The invasion diagnostic points in the opposite direction from the primary mixture point estimate and is also statistically inconclusive.

## Interpretation

This result does **not** establish that Stage B is stronger than Stage A. It also does **not** yet establish that Stage B regressed.

The strongest current signal is the primary mixture comparison: all domains point negative, and ALL/3H are close to excluding zero. However the 3H invasion diagnostic points positive, while HU direct is essentially unresolved. Therefore this is a borderline/mixed result rather than a safe basis for a training or architecture change.

Together with the prior evidence:

- weak fixed baselines showed no detectable Stage-B improvement/regression;
- decision-level policy drift was material, especially postflop/HU;
- contemporary cross-play does not show a Stage-B advantage and may be hinting at a small regression in the primary mixture metric.

The correct next action is **more read-only evaluation, not more training**.

## Higher-power confirmation gate

Repeat the exact contemporary cross-play on an independent fresh seed with 9000 scenarios:

```bash
SPINCORE_CROSSPLAY_SCENARIOS=9000 SPINCORE_CROSSPLAY_SEED=20260919 bash tools/run_lt2_checkpoint_crossplay.sh
```

Why 9000: it triples the initial sample while remaining far cheaper than another training block. If the current ALL/3H effect sizes persist with similar variance, the confidence intervals should tighten enough to resolve those domains. HU may remain wider and can be targeted separately only if needed afterward.

Do not combine or average results informally. Review the fresh-seed 9000-scenario report as an independent confirmation. If it again shows a coherent Stage-B disadvantage, pause same-regime training and inspect training dynamics/average-policy approximation before any extension. If it shows no disadvantage or reverses direction, treat checkpoint strength as unresolved and prioritize the faithful DeepCrusher benchmark rather than overfitting conclusions to checkpoint-vs-checkpoint cross-play.

## Stop condition

Do not resume training beyond iteration 7500 until the 9000-scenario fresh-seed confirmation is reviewed.
