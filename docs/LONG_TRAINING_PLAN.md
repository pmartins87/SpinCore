# SpinCore — Long-Training Plan

Status: **LT3 CONTINUATION PAUSED FOR ENS8 THROUGHPUT GATE**
Date: 2026-09-21

## Preserved baseline and continuation state

LT2 ENS8@8100 remains frozen and valid as the sealed-holdout-passed baseline.

Frozen LT2 identities:

- checkpoint SHA256:
  `a51dbbed71090e45f2c4f5db6297f72eab848990b38650702b436e2aa60ca4bf`;
- HU ENS8 sidecar SHA256:
  `c44b817f75304db352eedc33febbd180e6d038e0580c91be0b9befdbdb08f181`.

The interrupted LT3 H1 continuation has a matched durable checkpoint+sidecar at
iteration **8200**.  It is preserved for resume after the performance gate.

## Why continuation is still valid

The Stage-B forensic diagnosis did not find evidence that the mature HU
reservoir was poisoned.

Instead:

- the reservoir retained useful signal;
- HU fresh100 was insufficient at maturity;
- HU fresh400 recovered the signal in controlled refits;
- HU400 passed broad generalization;
- HU400 passed online-feedback validation;
- ENS8 stabilized mature HU current behavior;
- ENS8@8100 passed the frozen final holdout.

Therefore the historical learning state is not known to be contaminated.
Restarting from iteration 0 is not required by the evidence and would discard
millions of useful roots.

A from-zero corrected-schedule run remains an optional future research
experiment if we later want to measure path dependence, but it is not the
current production/research continuation plan.

## Current problem

The ENS8 implementation fits eight independent HU members sequentially.  This
is strategically correct but inefficient on the 32-thread Ryzen.

Before more long training:

1. benchmark exact-parity process-parallel member fitting;
2. include snapshot/serialization overhead;
3. choose the fastest exact layout that preserves all member states/losses;
4. derive a deterministic iteration target corresponding to approximately 24
   hours on the Ryzen, rounded to a checkpoint boundary;
5. resume from iteration 8200 to that precommitted endpoint rather than stopping
   merely because iteration 8600 was reached.

Iteration 8600 remains a required internal checkpoint for later comparison, but
the run must not inspect development-set results at 8600 and then make a
post-hoc continuation decision.  The ~24-hour endpoint is frozen before the
long run starts.

## Strategic invariants

- empirical SpinGo 3H/HU/blind/stack sampler;
- WTA chip-EV utility;
- SPNNIV1 frozen-control representation;
- repaired all-nonpositive regret fallback;
- separate 3H and HU domains;
- 2M reservoirs per memory/domain;
- 600 roots/iteration;
- 3H fresh100;
- HU ENS8 = 8 x fresh400;
- K4 off;
- LT2 artifacts read-only;
- LT2 final holdout retired and not reused;
- LT3 sealed holdout untouched until research choices are frozen.

## Immediate next gate

Run the Ryzen ENS8 parallel-fit matrix.

Only after exact parity and useful speedup are demonstrated should the
continuation resume from iteration 8200.

The final target iteration must be calculated from the measured optimized
throughput so the unattended block is approximately 24 hours long.
