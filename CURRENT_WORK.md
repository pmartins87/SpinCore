# SpinCore Current Work

Date: 2026-09-17
Status: **LT2 STAGE A PASS — 1.8M ROOTS — PRODUCTION CONCURRENT-FIT PARITY PASS — SHORT THROUGHPUT TUNING GATE BEFORE STAGE B**

## Active source of truth

Read before new compute:

- `docs/LT2_STAGE_A_REVIEW_20260916.md`
- `docs/LT2_CONCURRENT_FIT_PARITY_RESULT_20260916.md`
- `docs/LT2_PRODUCTION_CONCURRENT_PARITY_RESULT_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`

The active learning line is LT1 -> LT2. Preserve the finalized LT2 Stage A checkpoint and continue from it; do not restart LT0/LT1.

## Preserved source

LT2 Stage A is finalized at iteration 3000 / 1.8M roots:

`/home/rz9/spincore_lean_functional/runs/long_training_lt2/20260916_015559/checkpoint.pt`

SHA256:

`e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`

This checkpoint remains read-only for all tuning gates and for the eventual Stage B continuation.

## Evidence through Stage A

Stage A used 31 root workers, 8 parent Torch threads and vectorized batching. It completed with no swap use, minimum WSL MemAvailable about 10.67 GiB, process max RSS about 16.3 GiB, and a 2.236 GiB final checkpoint. Both Advantage reservoirs and the 3H AveragePolicy reservoir are saturated; HU AveragePolicy is 820,667/2M.

The weak-baseline learning curve remains positive overall and especially 3H. HU is still noisy/flat and is the principal strategic signal to re-check after HU AveragePolicy reaches capacity.

## Production concurrent fit — ADMITTED

The repeated fit benchmark gave a 1.312188x speedup for 8-thread concurrent 3H/HU Advantage fitting versus sequential 8-thread fitting, with exact same-thread model/loss/RNG parity.

The production iteration gate then passed exact semantic parity at iteration 3001:

- `semantic_parity=true`;
- `source_unchanged=true`;
- `first_difference=null`;
- reference wall 11.3933 s;
- production concurrent candidate wall 11.0136 s.

Production `--iteration-mode concurrent_fit` is therefore admitted. Sequential remains the default compatibility path.

## Why one more short tuning gate is justified

Before committing roughly half a day to Stage B, the user asked whether the Ryzen execution can be shortened further.

Stage-A fit profiles show that within each 100-step Advantage fit the optimizer/kernel work dominates: typical `optimizer_seconds` is about 3.1–3.3 s while batch construction is only about 0.18–0.22 s and reservoir sampling about 0.07 s. Further Python/batch micro-optimization therefore has little remaining room; the meaningful exact-semantics knobs are execution topology and checkpoint cadence.

Changing Torch thread count is not an exact-parity knob because different intra-op thread counts can change floating-point reduction order. Keep the admitted 8-thread fit contract for this line unless a separate numerical-drift study is explicitly opened.

Root worker count *is* an execution-only knob because root results are deterministically sorted and merged. It has not yet been production-shaped benchmarked for this SpinCore workload.

## Immediate finite gate — Stage B worker-count benchmark

Run exactly once:

`tools/benchmark_lt2_stage_b_worker_count.sh`

It tests workers `31,16,20,24,28`, always with:

- 8 parent Torch threads;
- vectorized batches;
- production concurrent-fit iterations;
- the exact preserved iteration-3000 checkpoint;
- one warmup + five timed iterations per worker count;
- each worker-count case in a fresh Python process;
- no checkpoint saves and no finalization.

Acceptance for changing away from 31 workers:

- exact final semantic signature across all worker counts;
- source checkpoint unchanged;
- >=3% median whole-iteration gain over workers=31.

If no candidate clears 3%, retain 31 workers and stop tuning this knob.

## Checkpoint cadence decision

Stage A checkpoint saves were roughly 45–55 seconds each. At Stage B's current every-250 cadence there would be 18 intermediate saves across 4500 iterations. Moving to every 500 would roughly halve that serialization count and likely save on the order of 8–10 minutes while doubling maximum restart loss from about one 250-iteration block to one 500-iteration block.

This is a modest but safe semantics-preserving gain. Final Stage B cadence will be set after the worker-count benchmark; do not remove the final checkpoint/finalization.

Memory telemetry and iteration logging are negligible compared with neural optimizer and tree work and should not be removed for speed.

## Stage B target after tuning gate

Continue the same LT2 state from iteration 3000 to approximately iteration 7500:

- +4500 iterations;
- +2.7M roots;
- 4.5M roots total;
- HU AveragePolicy expected to cross 2M near iteration 7.3k.

After Stage B, stop and review memory/swap, checkpoint behavior, actual throughput, 3H learning, and HU learning before any larger extension.

## Immediate user action

Do **not** start `run_long_training_lt2_stage_b.sh` yet. Pull `main`, run `bash tools/benchmark_lt2_stage_b_worker_count.sh`, and send `SpinCore_LT2_STAGE_B_worker_benchmark.json`. The benchmark is bounded and should take minutes, not hours. Also report the `windows_power_scheme=` line printed at the start; if Windows is on Balanced/Power Saver we can decide whether a host power-plan change is worthwhile before the long run.
