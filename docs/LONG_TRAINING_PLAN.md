# SpinCore — Long-Training Plan

Status: **LT2 TRAINING CLOSED / LT3 H1 TRAINING ACTIVE**
Date: 2026-09-21

## LT2

LT2 ENS8@8100 is frozen.

Its final strategic holdout passed and its deployment artifacts are preserved.
No further LT2 training is planned.

The LT2 final holdout is retired and must not be reused for LT3 research
decisions.

## LT3 Heavy H1 — active experiment

Purpose: test whether substantially more training beyond the mature LT2@8100
state produces a real development-set improvement without modifying the frozen
LT2 production baseline.

Preregistered H1:

- exact source checkpoint SHA256:
  `a51dbbed71090e45f2c4f5db6297f72eab848990b38650702b436e2aa60ca4bf`;
- exact HU ensemble sidecar SHA256:
  `c44b817f75304db352eedc33febbd180e6d038e0580c91be0b9befdbdb08f181`;
- source iteration: 8100;
- target iteration: 8600;
- additional iterations: 500;
- additional roots: 300,000;
- expected total roots: 5,160,000;
- 3H Advantage refit: fresh100;
- HU ENS8: 8 members x fresh400;
- HU optimizer steps per iteration: 3,200;
- K4: off;
- workers: 31;
- Torch threads: 8;
- checkpoint every 50 iterations.

## Guardrails

- LT2 source checkpoint and ensemble remain immutable;
- LT2 final holdout untouched;
- LT3 sealed holdout untouched;
- H1 is research-only and cannot be promoted directly;
- hard stop at iteration 8600;
- no automatic H2 continuation.

## After H1

Run the preregistered **development-set battery** and adjudicate H1 against the
frozen LT2@8100 baseline.

Only if development evidence supports continuation should H2 be designed.

The LT3 sealed holdout remains sealed until all research choices are frozen.

## Immediate command

```bash
bash tools/run_lt3_heavy_ens8_h1.sh
```

Expected completion sentinel:

`LT3_HEAVY_ENS8_H1_TRAINING_PASS`
