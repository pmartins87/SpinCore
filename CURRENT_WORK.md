# SpinCore Current Work

Date: 2026-09-20
Status: **ENS8 ONLINE PILOT REACHED 8100 — MECHANICAL PASS — POST-PILOT FORENSIC ADJUDICATION NEXT**

## Online pilot

The isolated ENS8 pilot completed iteration 8100.

Contract:
- source iteration 8000 remained read-only;
- +100 iterations / +60,000 roots;
- 3H unchanged single fresh100;
- HU 8 x fresh400 with predeclared ENS8_A seeds reused every iteration;
- raw Advantage average before unchanged lean regret matching;
- fit RNG isolated from sampled-policy RNG;
- K4 off;
- holdout untouched.

Final roots:
- THREE_HANDED 2,648,700;
- TRUE_HEADS_UP 2,211,300;
- total 4,860,000.

The run completed without fit/root/ensemble-size failure. HU member losses remained narrow and sampled action frequencies did not show a renewed catastrophic jam explosion.

This is not yet a strategic pass.

## Artifact contract

Current HU behavior at 8100 requires both:
- the ordinary checkpoint;
- `hu_ensemble_state.pt`.

The ordinary checkpoint alone stores only the last Advantage member.

AveragePolicy is ordinary/checkpoint-native because its training targets were generated from ENS8 behavior.

## Active gate

Read-only forensic adjudication:

- rebuild exact ENS8_A at source 8000;
- deterministic root drift ENS8 8000 → 8100;
- broad current-behavior EV versus ENS8 8000 and current 7600;
- AveragePolicy 8100 versus AveragePolicy 8000;
- Uniform / Passive / Jammer baselines.

No roots. Holdout sealed.

## Immediate action

```bash
bash tools/run_lt2_hu_ens8_post_pilot_forensic.sh
```

Wait for `LT2_HU_ENS8_POST_PILOT_FORENSIC_PASS`, then send
`SpinCore_LT2_hu_ens8_post_pilot_forensic.json`.

Do not train beyond 8100.
