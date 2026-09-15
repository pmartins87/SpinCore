# SpinCore Ryzen Optimization Policy

Status: **MANDATORY / CANONICAL**
Date: 2026-09-15

## Purpose

The user's Ryzen 9 is the project's primary heavy-compute machine. Any substantial SpinCore workload assigned to that machine must be designed and measured for that hardware rather than merely being functionally portable to it.

This policy is binding together with `AGENTS.md` and `docs/LEGACY_BASELINE_AND_QUALITY_POLICY.md`.

## Non-negotiable rules

1. **No long serial Ryzen jobs by default.** Before a job expected to take more than a short smoke/pilot is launched, identify the dominant CPU-bound work and parallelize independent work when doing so does not change poker semantics or learning objective.
2. **Measure the actual machine.** Do not assume that the maximum worker count is the fastest. Run one short worker-count benchmark on the Ryzen and select the best measured configuration. Re-benchmark only when the relevant execution architecture materially changes.
3. **Use the DeepPot precedent.** On this same 32-logical-thread Ryzen, the proven production pattern was many independent worker processes with one Torch/OMP/MKL thread per worker, normally leaving one logical CPU for the parent/OS. SpinCore should reuse that pattern where roots/tasks are independent.
4. **Avoid oversubscription.** Worker processes use one Torch/OpenMP/BLAS thread each unless a measured benchmark demonstrates a better configuration. Parent fitting/inference thread counts are tuned separately.
5. **Optimize execution, not strategy semantics.** Parallelization may change scheduling but must not silently change the game model, scenario distribution, legal actions, terminal utility, state representation, or the mathematical Deep-CFR operation being approximated. The parent remains authoritative for scenario sampling and reservoir state.
6. **Persistent pools for iterative training.** Process-spawn/model-load overhead must be amortized across iterations rather than paid per root or per iteration when avoidable.
7. **Checkpoint before interruption.** Long training must remain resumable. If an optimization is introduced while a serial run is active, stop only after a durable checkpoint when practical, then resume from that checkpoint under the optimized execution path.
8. **CPU utilization is a diagnostic, not the objective.** High utilization is desirable only when it reduces time-to-useful-training without harming correctness. Do not manufacture 100% CPU by adding waste. Select the configuration by measured useful throughput (e.g. root-collection seconds / roots), memory safety and stability.
9. **Profile the next bottleneck.** Once root collection is accelerated, if neural fitting, serialization, checkpoints, reservoir operations or another phase becomes dominant, optimize that phase specifically. Do not keep increasing root workers past the point of diminishing returns.
10. **Record the selected profile.** The chosen worker count and benchmark results belong in `runs/worker_benchmark/` and the current project state so a future chat/session does not revert to an unoptimized serial command.

## Current implementation

The original 2026-09-15 Ryzen pilot intentionally used two Torch threads and serial root collection. It consumed ~193% process CPU, which is approximately two fully occupied logical CPUs and explains the ~6-7% total CPU seen on a 32-thread machine. It was a mechanics/throughput smoke, not an acceptable production configuration.

The optimized functional path parallelizes **independent Deep-CFR advantage roots** with a persistent `ProcessPoolExecutor`. Workers use one Torch/OMP/MKL/OpenBLAS/NumExpr thread each. Scenarios are sampled by the parent, workers return advantage samples, and the parent merges them into the authoritative reservoir in deterministic root order.

Relevant files:

- `python/spincore/lean_parallel.py`
- `python/spincore/lean_functional_training.py`
- `tools/benchmark_lean_ryzen_workers.sh`
- `tools/run_lean_functional_first_training.sh`
- `tools/resume_latest_lean_training_optimized.sh`

The one-time local benchmark compares 1, 8, 16, 24 and 31 root workers (subject to machine capacity), skips configurations that fail, and selects the successful configuration with the lowest fitted-policy iteration root-collection time. The selected count is persisted to `runs/worker_benchmark/selected_workers.txt` and is automatically consumed by the production/resume scripts.

## Quality invariant

Ryzen optimization is not permission to weaken the solver. The optimization target is **more strategically useful samples per wall-clock hour for the same intended training algorithm**. When an optimization would materially approximate or change the learning problem, it must be treated as an architecture/quality decision under the legacy-first policy rather than as a performance tweak.
