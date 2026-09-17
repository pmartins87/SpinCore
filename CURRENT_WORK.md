# SpinCore Current Work

Date: 2026-09-16
Status: **LT2 STAGE A PASS — 1.8M ROOTS — RESOURCE GATE HEALTHY — LEARNING CURVE POSITIVE OVERALL/3H — HU STILL NOISY — CONCURRENT-FIT SCREEN NEXT**

## Active source of truth

Read these before new compute:

- `docs/LT2_STAGE_A_REVIEW_20260916.md`
- `docs/LONG_TRAINING_PLAN.md`
- `docs/LT1_COMPLETION_REVIEW_20260916.md`
- `docs/LT1_FIT_BENCHMARK_RESULT_20260916.md`

The active long-training line is **LT1 -> LT2**. Do not restart from LT0/LT1, do not rerun the closed 1/2/4/8/16-thread matrix, and do not treat weak-baseline tests as a final-strength verdict.

## Preserved checkpoints

LT0 calibration:

- 200 iterations / 120k roots;
- `/home/rz9/spincore_lean_functional/runs/lean_first_training/20260915_131133/checkpoint.pt`.

LT1 production-shaped milestone:

- 2000 iterations / 1.2M roots;
- `/home/rz9/spincore_lean_functional/runs/long_training_lt1/20260915_181249/checkpoint.pt`;
- SHA256 `beef9bee9439de8d9153450d190e62b0388a678b0778c4185108361f626b4337`.

LT2 Stage A current continuation checkpoint:

- 3000 iterations / **1.8M roots total**;
- `/home/rz9/spincore_lean_functional/runs/long_training_lt2/20260916_015559/checkpoint.pt`;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`;
- finalized and preserved as the source for the next stage.

## LT2 Stage A — PASS

Stage A continued the exact LT1 learning state for 1000 iterations / 600k additional roots with 31 root workers, 8 parent Torch threads and vectorized batching.

Resource evidence:

- trainer wall scope: 10,547.402 s = 2h55m47s;
- checkpoint final size: 2.235843 GiB;
- final checkpoint save: 54.468 s;
- minimum observed WSL `MemAvailable`: about 10.67 GiB;
- maximum swap used: 0 GiB;
- process max RSS: about 16.3 GiB;
- exit status 0 / `LT2_STAGE_A_PASS`.

Reservoir state at iteration 3000:

- 3H Advantage seen: 30,645,342 — reservoir saturated;
- HU Advantage seen: 24,655,892 — reservoir saturated;
- 3H AveragePolicy seen: 2,216,393 — **crossed the 2M capacity during Stage A**;
- HU AveragePolicy seen: 820,667 — still below capacity.

Checkpoint growth fell sharply after the 3H policy reservoir saturated: roughly 33–35 MiB per 100 iterations before the crossing, then roughly 9–11 MiB per 100 near iteration 3000. There is no runaway memory or serialization signal.

## Learning curve — useful improvement continues, mainly 3H

The same fixed-seed 1000-scenario weak-baseline diagnostic was run on 120k, 1.2M and 1.8M checkpoints.

Uniform-legal overall cEV:

- +10.836 -> +15.873 -> **+17.015** chips/hand.

Uniform-legal 3H cEV:

- +2.407 -> +8.898 -> **+12.769**.

Passive-caller overall cEV:

- -2.402 -> -0.553 -> **-0.064**.

Jammer overall cEV:

- -5.498 -> -2.719 -> **-1.298**.

Jammer 3H cEV:

- +2.295 -> +4.045 -> **+7.422**.

Interpretation: additional training is still producing useful poker improvement overall and particularly in 3H. The line is not stalled globally.

HU is the unresolved signal. From LT1 to LT2-A the fixed-seed HU point estimates were flat/slightly worse across the three weak families. The intervals remain wide, and the HU AveragePolicy reservoir is only at 820,667/2M, so this is **not** evidence that HU has reached its ceiling. It is a reason to track HU explicitly at the next meaningful checkpoint.

The comparison JSON itself notes that checkpoint deltas are descriptive; it does not provide a dedicated paired CI for checkpoint-vs-checkpoint change.

## Ryzen execution profile

Current proven production profile:

- 31 root workers;
- worker Torch/OpenMP/BLAS threads = 1;
- parent Torch threads = 8;
- `batch_mode=vectorized`.

Stage A also confirmed the user's observation that average CPU utilization remains low relative to 32 logical CPUs. The two domain Advantage fits are currently serial and remain the dominant per-iteration phase. We will not change execution order blindly just to make Task Manager look full.

## Immediate finite gate — concurrent-fit screen

Before committing roughly another half day of Ryzen time to Stage B, run exactly one read-only benchmark:

`tools/benchmark_lean_lt2_concurrent_fit.sh`

It tests 3H/HU Advantage optimizer overlap at 4 and 8 Torch threads against the proven sequential-8 baseline. Model construction/reset remains sequential so the Torch RNG isolation contract cannot race. A candidate is interesting only if:

- same-thread sequential/concurrent final model hashes match exactly for both domains;
- final losses match exactly;
- per-domain batch RNG states match exactly;
- source checkpoint remains byte-identical;
- median combined fit wall time improves by at least 5% versus sequential 8-thread fitting.

This benchmark does **not** start training and cannot by itself authorize concurrent production execution.

If it fails to produce >=5% exact-parity gain, close the concurrency branch and proceed with the existing 31/8/vectorized path. If it passes, implement one full-iteration semantics-preserving candidate and prove exact iteration parity before Stage B.

## Next training milestone after the fit screen

If the execution path is cleared, continue the same LT2 checkpoint to approximately **iteration 7500** (+4500 iterations / +2.7M roots, 4.5M roots total). At the Stage-A HU policy-sample rate, the HU AveragePolicy reservoir should reach 2M around iteration 7.3k, making ~7500 the natural next bounded milestone.

At that point re-evaluate:

- resource/checkpoint behavior with all four 2M reservoirs effectively saturated;
- 3H learning continuation;
- HU learning versus uniform, passive caller and jammer;
- direct DeepCrusher benchmark once the faithful oracle is ready.

## DeepCrusher

DeepCrusher remains a mandatory future strength/acceptance reference, not the training teacher and not a gate that should interrupt early serious training. Continue the faithful OpenPPL oracle work in parallel. Do not use a simplified imitation for canonical claims.

## Immediate user action

Run the concurrent-fit benchmark only. **Do not start LT2 Stage B yet.** Send its `report.json` back for the integration/continue decision.
