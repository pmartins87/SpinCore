# SpinCore Current Work

Date: 2026-09-18
Status: **LT2 STAGE B PASS — 4.5M ROOTS — HU-JAMMER NEGATIVE — K4 MECHANICS PASS — FORENSIC FIRST RUN TERMINATED — LIGHTWEIGHT-POLICY FIX READY — RERUN NEXT**

## Active source of truth

Read before new compute:

- `docs/LT2_HU_PREFLOP_BOARD_AVERAGING_SMOKE_RESULT_20260918.md`
- `docs/LT2_STAGE_A_B_FIRST_DIVERGENCE_FORENSIC_20260918.md`
- `docs/LT2_STAGE_A_B_FIRST_DIVERGENCE_RESOURCE_FAILURE_20260918.md`
- `docs/LT2_HU_PREFLOP_BOARD_ONLY_AVERAGING_RESULT_20260918.md`
- `docs/LT2_HU_PREFLOP_TARGET_ESTIMATOR_BUDGET_RESULT_20260917.md`
- `docs/LT2_WEAK_BASELINE_VARIANCE_RESULT_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`

Preserve Stage A and Stage B. Do not continue root training beyond iteration 7500.

## Preserved checkpoints

Stage A: iteration 3000 / 1.8M roots, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Confirmed practical failure

Powered 30k weak-baseline gate:

- Stage B HU Jammer raw chip EV `-5.141`, simultaneous family-wise 95% CI `[-9.078,-1.204]`;
- Stage-B-minus-Stage-A HU-Jammer paired delta `-1.682`, simultaneous six-claim interval approximately `[-3.143,-0.222]`.

Long root scaling remains paused.

## Estimator mechanism evidence

HU-preflop conditional decomposition:

- future-board variance: **65.88%**;
- opponent-hand posterior variance: **26.10%**;
- residual exact-level-1 opponent-action variance: **1.74%**;
- current-model MSE to conditional mean: **6.27%**.

Board-only K4 is the measured estimator compute elbow.

This is still estimator evidence, not proof that target noise caused the A->B deployed-policy regression.

## K4 mechanics smoke — PASS

Corrected smoke on the same 8 deterministic prospective HU roots:

- samples per arm: `273`;
- preflop samples: `22`;
- K4 changed `21/22 = 95.45%` preflop targets;
- postflop samples: `251`;
- postflop target differences: `0`;
- K1 nodes: `1,229`;
- K4 nodes: `4,765`;
- measured node multiplier: **3.8771x**;
- maximum preflop target absolute delta: `0.592`.

All invariants passed:
- same root jobs;
- same sample count/order/identity;
- canonical postflop targets unchanged;
- canonical RNG progression preserved;
- no optimizer steps;
- no training-memory writes;
- Stage-B source unchanged.

The K4 implementation is mechanically admissible.

**It is not yet scientifically admitted as a training fix.**

## Why no training yet

The weak-baseline evaluator uses the stored **AveragePolicy**.

K4 changes Advantage target generation, which could influence future behavior and AveragePolicy data, but the causal chain has not been demonstrated.

The immediate question is therefore:

**Where does Stage B's deployed AveragePolicy first diverge from Stage A on the exact paired HU evaluation trajectories, and which divergence contexts contribute the negative B-A chip EV?**

## First forensic run — resource termination

The first full forensic run was terminated by the OS/WSL before producing a Python traceback or report. The original design spawned 31 workers, and every worker deserialized both **full** Stage-A and Stage-B training checkpoints through `LeanFunctionalAgent.from_checkpoint`. That redundantly loaded large reservoirs and both domains even though the audit needs only the HU AveragePolicy.

The resource design has been corrected:

- full checkpoints are read once in the parent, using mmap when supported;
- only TRUE_HEADS_UP AveragePolicy weights are extracted;
- workers load tiny policy-only snapshots and never reservoirs;
- default workers reduced from 31 to 16;
- forensic sampling, pairing and attribution semantics are unchanged.

The trailing `resource_tracker` semaphore warning is treated as a consequence of abrupt multiprocessing shutdown, not as the cause.

## Active forensic gate

Run:

```bash
bash tools/run_lt2_stage_a_b_first_divergence.sh
```

The audit:

- reuses the already-seen forensic seeds `20260920..20260925`;
- uses 5,000 full-sampler scenarios per seed by default;
- retains HU scenarios;
- compares Stage A and Stage B AveragePolicy directly;
- includes UNIFORM_LEGAL, PASSIVE_CALLER and JAMMER;
- keeps scenario/deal/opponent/seat/random streams paired until first hero-policy divergence;
- partitions first divergence into:
  - NO_DIVERGENCE;
  - PREFLOP_ROOT;
  - PREFLOP_FACING_ALL_IN;
  - PREFLOP_OTHER;
  - FLOP;
  - TURN;
  - RIVER;
- decomposes total B-A chip EV into additive first-divergence contributions.

The already-seen seeds are diagnosis/design data only.

Reserve `20261001..20261006` as untouched holdout seeds for future candidate acceptance.

## Decision after forensic audit

- If the negative regression is mainly postflop, K4 is not the primary fix.
- If it is preflop but not aligned with K4-sensitive states, K4 remains unproven.
- If a resolved negative contribution is concentrated in HU-preflop/FACING_ALL_IN and Stage B shifts action mass in the same problematic direction seen in the low-noise target diagnostics, run one final Advantage/target overlay before considering K4 training.
- Jammer-only evidence is not enough; cross-baseline consistency matters.

DeepCrusher remains deferred.

## Immediate user action

Pull current `main` and run `bash tools/run_lt2_stage_a_b_first_divergence.sh`.

Wait for `LT2_STAGE_A_B_FIRST_DIVERGENCE_FORENSIC_PASS` or the first error.

Then send `SpinCore_LT2_stage_a_b_first_divergence.json`.

Do not start any K4 or long-training continuation.
