# SpinCore — LT3 Research Status

Date: 2026-09-21
Status: **H1 READY TO LAUNCH**

## Frozen baseline

Production baseline: LT2 ENS8@8100.

LT3 source is read-only and hash-locked.

## H1

- 8100 -> 8600;
- +500 iterations;
- +300,000 roots;
- ENS8 x fresh400 unchanged;
- 3H fresh100 unchanged;
- 31 workers;
- Torch threads 8;
- checkpoint every 50 iterations;
- research-only output directory:
  `runs/lt3_heavy_ens8_h1/<timestamp>/`.

Expected wall time from the measured LT2 8000->8100 pilot is approximately
18 hours if throughput remains similar. This is an estimate, not a deadline.

## Start command

```bash
cd ~/spincore_lean_functional &&
git pull --ff-only origin main &&
bash tools/run_lt3_heavy_ens8_h1.sh
```

Successful completion sentinel:

`LT3_HEAVY_ENS8_H1_TRAINING_PASS`

Stop at 8600. Do not extend automatically.
