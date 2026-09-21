# SpinCore Current Work

Date: 2026-09-21
Status: **LT3 21H PARALLEL CONTINUATION AUTHORIZED — 8200 -> 9105 / OPENHOLDEM PAUSED**

## Strategic baseline

LT2 ENS8@8100 remains the frozen production baseline.

Its checkpoint, HU ENS8 sidecar, final holdout evidence and deployment artifacts
must remain read-only.

The previous OpenHoldem productionization work is preserved but **paused**.
No OpenHoldem host inspection, DLL installation or table testing is required
while LT3 training is the user's active priority.

## Primary active lane — LT3 continuation from preserved 8200

LT2 ENS8@8100 remains the frozen sealed-holdout-passed baseline.

The interrupted LT3 continuation has a durable matched checkpoint+sidecar at
iteration 8200.  It is preserved and remains eligible for continuation.

Why we are **not** restarting from zero:

- the Stage-B reservoir-poisoning hypothesis was not supported;
- controlled fresh refits showed usable signal remained in the mature reservoir;
- the material defect was insufficient HU fitting (fresh100), repaired by HU400;
- HU400 passed structural, broad and online-feedback validation;
- ENS8 then stabilized the mature HU current behavior and the 8100 candidate
  passed the pre-registered sealed holdout.

Thus there is no evidence that the accumulated 0..8100 learning state is
invalid.  A clean-from-zero run would be a separate research arm, not a required
repair.

Immediate engineering issue: the current ENS8 fit implementation is
unnecessarily sequential.  Before resuming 8201+, benchmark process-parallel
member fitting and require exact state/loss parity.

Operational scheduling constraint: after the performance matrix, do not default
to a short 8200->8600 run that is likely to finish while the user is unavailable.
Use the measured optimized iteration wall time to precommit a block of
approximately 24 hours, rounded to a checkpoint boundary.  Preserve 8600 as an
internal snapshot for comparison, but continue automatically to the predeclared
24-hour endpoint without looking at development outcomes mid-run.


## OpenHoldem deployment lane — PAUSED

Completed before pause:

- observable E2E tracker PASS;
- native C++ tracker PASS;
- native tracker + frozen inference shadow engine PASS;
- Windows x64 shadow user-DLL build PASS;
- Windows mock-host LoadLibrary/ABI gate PASS.

Paused next step:

- inspect the actual OpenHoldem host architecture.

No deployment work is needed now.

## Immediate action

With the sequential continuation stopped and 8200 preserved, run:

```bash
bash tools/run_lt3_hu_ens8_parallel_fit_benchmark.sh
```

Expected sentinel:

`LT3_HU_ENS8_PARALLEL_FIT_BENCHMARK_PASS`

Do not restart H1 until this benchmark is adjudicated.


## ENS8 parallel matrix incident

The first parallel-fit matrix (V1) was terminated after preflight.  Source
inspection found that V1 serialized the complete 2M-sample HU Advantage
reservoir and then deserialized that Python object graph independently in each
fit subprocess.  That design is not acceptable for the Ryzen/WSL memory
envelope and is superseded regardless of the exact OS termination reason.

V2 replaces per-worker reservoir copies with one compact mmap mirror:

- observations, legal masks, targets and weights are stored once;
- workers open the mirror read-only;
- each member reproduces the historical Python-random sample-index stream;
- the canonical sequential path remains the exact parity reference;
- full Python reservoirs are released before worker pools are created;
- worker RSS is recorded;
- steady-state fit speedup excludes one-time pool/mirror initialization but both
  one-time costs are reported separately.

Do not rerun V1.  Run only the V2 matrix.


## ENS8 parallel matrix V2 — PASS

Measured on the Ryzen against the canonical 8-thread sequential ENS8 fit:

- sequential HU ENS8 fit wall: 114.824 s;
- 2x8: 83.143 s, 1.381x, exact state/loss parity;
- 4x8: 73.027 s, 1.572x, exact state/loss parity — **selected**;
- 8x8: 3777.466 s, exact but severe oversubscription collapse;
- lower-thread layouts (4x4, 8x4, 8x2) were faster but failed exact
  state/loss parity and are rejected.

Important: 1.572x is the measured steady-state speedup of the **HU ENS8 fitting
phase**, not yet the whole training iteration.  A short full-iteration
integration benchmark is required before calculating the ~24-hour block target.

Next gate: integrate persistent mmap + 4x8 fitting into the LT3 continuation
path and measure end-to-end iteration wall time without generating a long run.


## 8200 end-to-end integration gate — PASS

The fit-only matrix is no longer sufficient to authorize the ~24-hour run.
The selected 4x8 fitter is now wired into a disposable end-to-end gate from the
preserved 8200 checkpoint.

The gate runs in isolated processes:

- one sequential control iteration: 8201;
- three parallel 4x8 disposable iterations: 8201..8203;
- exact semantic fingerprint comparison after the shared 8201 iteration;
- authoritative reservoir-write indices/sample digests;
- model and optimizer states;
- RNG states and counters;
- sampler state;
- HU ensemble member states;
- whole-iteration wall time.

The source 8200 checkpoint+sidecar remain read-only.  Only after this gate
passes will the ~24-hour target be frozen.


## 21-hour continuation contract — FROZEN

The 8200 end-to-end gate passed with exact semantic parity.

Measured on the same disposable iteration 8201:

- sequential whole iteration: 123.932 s;
- parallel 4x8 whole iteration: 80.992 s;
- whole-iteration speedup: 1.5301658x;
- parallel median over 8201..8203: 81.108 s;
- historical checkpoint cost: 101.099 s every 50 iterations;
- planning time including checkpoint amortization: 83.130 s/iteration.

The next long block is frozen before training starts:

- source: durable matched checkpoint+sidecar @8200;
- target: 9105;
- additional iterations: 905;
- new roots: 543,000;
- projected total wall: 20.998 h;
- checkpoint every 50;
- raw @8600 checkpoint+sidecar preserved automatically;
- HU fit: exact-parity 4x8;
- no LT2 final-holdout access;
- no LT3 sealed-holdout access.

Do not change target based on intermediate results. The earlier 9250/24.35 h plan is superseded.
