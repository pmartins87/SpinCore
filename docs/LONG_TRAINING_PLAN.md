# SpinCore — Long-Training Plan

Status: **LT3 8200 -> 9105 EXACT-PARITY CONTINUATION RUNNING**
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

Iteration 8600 remains a required internal raw checkpoint for later comparison. The active run continues without adjudication to the frozen 9105 endpoint. Evaluation of 8600 requires finalizing AveragePolicy on a derived copy after the run, because the raw in-run milestone is deliberately not finalized.

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


## Matrix V1 memory incident

The first process-parallel matrix was terminated after preflight.  Its design
copied/deserialized the full 2M-sample HU Advantage reservoir into each member
process and is therefore superseded as memory-unsafe.

The replacement V2 uses a single compact mmap reservoir mirror shared read-only
by fit workers.  The mirror is intended to persist across training iterations;
a production continuation must update only reservoir slots actually replaced,
not rebuild/copy the full reservoir every iteration.

V2 must pass exact final-state and exact final-loss parity before it can be used
for the 8200 continuation.


## ENS8 parallel matrix V2 result — PASS

The exact-parity matrix selected **4 concurrent members x 8 Torch threads**.

Measured:

- sequential HU ENS8 fit: 114.824 s;
- 4x8 HU ENS8 fit: 73.027 s;
- steady-state HU fit speedup: 1.5723587x;
- all eight final model states exact;
- all eight final losses exact.

Do not interpret 1.572x as the final whole-training speedup.  Root collection,
3H fitting, strategy sampling, mmap maintenance, checkpoints and other iteration
work remain outside this fit-only ratio.

Before freezing the ~24-hour target, run a bounded full-iteration throughput
gate with the selected 4x8 implementation and derive the target from measured
end-to-end wall time.


## 8200 end-to-end gate result — PASS

The selected 4x8 fitter passed full semantic parity on the same disposable
iteration 8201.  Source 8200 artifacts remained unchanged.

Measured:

- sequential iteration 8201: 123.932 s;
- parallel iteration 8201: 80.992 s;
- whole-iteration speedup: 1.5301658x;
- parallel 8201..8203 median: 81.108 s;
- checkpoint amortization: 2.022 s/iteration;
- planning time: 83.130 s/iteration.

The user-adjusted frozen continuation target is iteration **9105**, or +905 iterations from
8200. Using the same gate planning rate, the projected wall time is 20.998 h including historical end-block overhead. The earlier 9250/24.346 h target is superseded.

Iteration 8600 is preserved automatically as a raw internal checkpoint+ENS8
sidecar but does not stop or alter the precommitted run.


## Post-run strength adjudication

After the 9105 PASS, do not automatically launch more training.

First compare 8100, a derived finalized copy of 8600, and finalized 9105 on the
development battery.  Then use the external DeepCrusher benchmark.

The canonical external opponent remains frozen DeepCrusher R8 v22.  The
benchmark match engine and exact-action bridge exist, but the DeepCrusher
decision oracle must still satisfy DC0 source/runtime fidelity before a
canonical strength claim.

No iteration-to-strength extrapolation is authorized.  DC2, not iteration
count, defines external qualification: positive paired chip EV with the overall
95% CI lower bound above zero and no major-domain collapse.
