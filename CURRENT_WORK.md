# SpinCore Current Work

Date: 2026-09-15
Status: **FUNCTIONAL PATH VALIDATED — RYZEN PARALLEL OPTIMIZATION IMPLEMENTED, LOCAL WORKER BENCHMARK NEXT**

## Goal

Make the multi-year DeepSpin project actually work as SpinCore. Preserve mature legacy knowledge; replace only components with a concrete correctness, learning-quality, or compute-efficiency reason.

## First functional SpinCore — decisions closed

- **Scenario distribution:** restored legacy real SpinGo model: 3-handed + true HU, nine blind levels from 10/20 to 100/200, separate empirical blind weights, blind-conditioned stack distributions, 1500 total chips, random dealer/live/dead seats.
- **Payout scope:** one winner-take-all policy family first, reused initially across payout variants. No separate 70/30, 50/30/20 training now.
- **Utility:** chip EV remains the strategic objective. Legacy state-dependent `chip_delta / current_bb` target scaling is replaced by one global positive scale `chip_delta / 1500`. State inputs such as stack/BB and pot/BB remain BB-relative because that geometry is strategically meaningful.
- **Representation:** compact SPNNIV1 exact-state-derived neural boundary. Do not restore the full legacy 292-float vector.
- **Action scope:** preserve the mature DeepSpin seven labels with their actual historical context semantics. Preflop uses 2BB open, 2.5BB+ isolation, 5BB+ 3-bet families and limp/multi-raise restrictions; postflop uses 33/50/75/100% pot-after-call plus legacy near-all-in collapse.
- **Deep CFR traversal:** standard external sampling (`exact_opponent_levels=0`).
- **Average-policy memory:** ordinary sampled game trajectories after advantage fitting, matching mature DeepSpin and avoiding R7.5 exact-expansion blow-up.
- **Regret fallback:** fitted advantage networks with all legal outputs <= 0 use masked softmax over raw advantages; uniform is reserved for genuinely untrained initialization.
- **Domains:** separate `THREE_HANDED` and `TRUE_HEADS_UP` brains with realistic per-domain sampling.

## Ryzen optimization is mandatory

`docs/RYZEN_OPTIMIZATION_POLICY.md` is canonical. Substantial workloads on the user's 32-logical-thread Ryzen must be optimized for that machine before long execution. The proven DeepPot precedent is many independent worker processes with one Torch/OMP/MKL thread per worker, normally leaving one logical CPU to the parent/OS.

The original functional pilot and the first 120k script revision were serial root collectors with two Torch threads. The pilot reported about **193% process CPU**, i.e. approximately two fully used logical CPUs, explaining the user's ~7% Task Manager total CPU reading. That serial configuration was useful for functional measurement but is **not accepted as the production Ryzen configuration**.

SpinCore now parallelizes only independent **advantage-root collection** with a persistent process pool. Each worker uses one Torch/OpenMP/BLAS thread. The parent remains authoritative for empirical scenario sampling and the global reservoirs and merges returned samples in root order. Poker rules, state representation, legal-action semantics, utility and Deep-CFR node logic are unchanged by this execution optimization.

Relevant optimization files:

- `python/spincore/lean_parallel.py`
- `tools/benchmark_lean_ryzen_workers.sh`
- `tools/run_lean_functional_first_training.sh`
- `tools/resume_latest_lean_training_optimized.sh`

The parallel path has passed GitHub CI with two workers. The actual Ryzen must now benchmark 1/8/16/24/31 workers once. Failed oversubscribed configurations are skipped; the fastest successful root-collection profile is persisted to `runs/worker_benchmark/selected_workers.txt` and automatically reused by future substantive/resume scripts.

## Ryzen pilot evidence — 2026-09-15

The pilot completed 5 iterations / 1000 roots in **28.8 s trainer wall time**, peak RSS about **389 MB**, using ~193% CPU. It produced 545 3H + 455 HU roots, 68,860 3H nodes + 42,323 HU nodes, 12,598 3H advantage samples + 8,906 HU, and 1,081 3H strategy samples + 390 HU. Real blind sampling reached late HU levels through 80/160.

The finalized checkpoint passed 100-hand offline self-play with no illegal action. That tiny policy was still excessively shove/fold-heavy (310 preflop decisions, only 15 flop, no turn/river) and is not strategy-quality evidence.

## First substantive profile

Once the selected Ryzen worker count is known, the same intended training profile is:

- 200 iterations;
- 600 roots/iteration = 120,000 roots;
- about 327 3H + 273 HU roots/iteration;
- about 1527 advantage traversals/iteration, nearly the mature DeepSpin 1536;
- about 255 sampled-policy episodes/iteration, nearly the mature DeepSpin 256;
- external sampling (`exact_opponent_levels=0`);
- reservoir capacity 100,000;
- 50 advantage optimizer steps/iteration, batch 256;
- 400 final AveragePolicy optimizer steps, batch 256;
- checkpoint every 5 iterations;
- 5,000-hand offline self-play after training.

The checkpoint is resumable and may be extended rather than discarded.

## Active serial run started before optimization

If the user already started the older serial 120k script, do **not** waste completed work. Prefer to stop immediately after the next printed `CHECKPOINT .../checkpoint.pt` line, then pull the optimized code, run the short worker benchmark, and resume the newest checkpoint with `tools/resume_latest_lean_training_optimized.sh`. Stopping between periodic checkpoints can lose up to the unsaved iterations since the prior checkpoint but does not corrupt the last durable checkpoint.

## Historical failure lesson

The earlier DeepSpin trained for roughly three months and still made gross mistakes. Do not answer bad play with 'train longer' before checking game/evaluator semantics, state representation, sampling, regret behavior, action mapping and inference parity. Known historical defects included board-only made-hand interpretation, a uniform all-nonpositive regret fallback and architecture/runtime drift.

## Do not do

- Do not resume Dense-reference i3-i5 merely to complete an old matrix.
- Do not use fixed-10/20 PF0-PF4 evidence as the final global selector.
- Do not restart the old R8/gate chain.
- Do not create payout-specific trainings now.
- Do not reopen representation/action tournaments without concrete play evidence.
- Do not knowingly run long serial/low-utilization workloads on the Ryzen when independent work can be parallelized safely.

## Immediate next milestone

Stop any pre-optimization serial substantive run at the next durable checkpoint, run the one-time Ryzen worker-count benchmark, then resume that same checkpoint using the selected optimized worker count. Only after measured parallel throughput is known should the remaining duration of the first substantive training be accepted.
