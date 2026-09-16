# LT1 neural-fit optimization — 2026-09-16

Status: **PHYSICAL RYZEN PASS — winner frozen at 8 threads + vectorized batches; LT2 Stage A next.**

See `LT1_FIT_BENCHMARK_RESULT_20260916.md` for the complete physical result and decision.

## Evidence and decision

The full LT1 report attributes 71.08% of measured time to Advantage fitting. Current batching decodes 1,024 SPNNIV1 payloads to Python dataclasses/lists every optimizer step. That identified a concrete avoidable cost. The prior thread benchmark had used batch 256 and 30k reservoirs, whereas LT1 uses batch 1024 and 2M capacity, so a physical benchmark at the real LT1 workload was required before promotion.

Legacy preflight performed against the supplied Tentativas anteriores de SpinGo.zip: deepspin/buffers.py (array-backed sample_batch), trainer.py (array-to-tensor fit), and networks.py. Preserve the array batching principle, not its different 292-feature MLP or sampling-with-replacement semantics. Current Python random.sample, LT1 wire bytes, reservoirs, optimizer, weighted loss, network reset, 32-token padded GRU history, ten-slot external legal mask, sampler, actions and utility remain unchanged.

## Implemented

- Optional vectorized batch construction: `python/spincore_nn/lean_batch.py`.
- Reference path remains available for parity/regression checks; historical V2 paths unchanged.
- Fit timings distinguish sample selection, batch assembly and optimizer work.
- Trainer accepts `--batch-mode reference|vectorized`, explicitly applies the requested parent thread count and records it.
- Final report includes the finalized checkpoint save metric and measured wall scope. Terminal prints a compact final summary; complete history stays in report.json.
- No concurrent-domain fit introduced: the physical measurement showed a worthwhile win without adding that scheduling/RNG complexity.

## Physical benchmark — COMPLETE

The bounded benchmark was executed on the user's Ryzen against the real finalized LT1 checkpoint:

```text
source = /home/rz9/spincore_lean_functional/runs/long_training_lt1/20260915_181249/checkpoint.pt
source iteration = 2000
source SHA256 = beef9bee9439de8d9153450d190e62b0388a678b0778c4185108361f626b4337
benchmark code commit = 6d9901366a9281a584e3f03703b6bb4efaf364b9
benchmark report SHA256 = c63596d505c24fea51e260237ce33495f1d02f0c23efd2cbc222388a034957b1
```

Ten finite cases compared 1/2/4/8/16 threads under reference and vectorized batch construction, with full checkpoint-configured fit steps and batch size. Exact tensor parity and same-thread final model/RNG parity were enforced by the benchmark.

The baseline 8-thread/reference case took `7.663904482004 s` combined across 3H and HU. The fastest case was 8-thread/vectorized at `6.5824409390043 s`, yielding `1.1642952140430884x` fit throughput and a 14.11% reduction in fit time. The frozen 5% promotion threshold was therefore passed.

The vectorized thread ranking was:

```text
8 threads   6.58244 s
16 threads  8.47159 s
4 threads   9.11097 s
2 threads   9.16187 s
1 thread   14.88826 s
```

The gain is mainly batch construction: at 8 threads, 3H batch assembly fell from about 0.595 s to 0.167 s and HU from about 0.595 s to 0.164 s. Optimizer time changed only modestly. This is the intended outcome: remove Python-side preparation overhead without changing the learning algorithm.

## Resume integration — COMPLETE

After selection, the benchmark restored LT1 model/RNG/counters and executed exactly one disposable resumed full iteration using the winner and the existing 31-worker root profile.

```text
iteration = 2001
3H roots = 327
HU roots = 273
total roots = 600
execution mode = parallel_31x1 in both domains
source_unchanged = true
status = LT1_FIT_BENCHMARK_PASS
```

The disposable iteration was not saved or promoted. The original LT1 checkpoint remains the continuation source.

## Frozen next-stage execution profile

```text
root workers = 31
worker Torch/OpenMP/BLAS threads = 1
parent Torch threads = 8
batch_mode = vectorized
```

Do not rerun the 1/2/4/8/16 matrix merely to seek a better-looking result. The optimization screen is closed.

## Local validation retained

- 13 targeted pytest cases passed: wire/tensor parity for batches 1/37/1024, exact model and Adam-state parity for Advantage/Policy updates, malformed payload rejection, historical model and training tests.
- Solver compiled in Release (existing compiler warnings remain).
- End-to-end tiny finalized checkpoint loaded by benchmark, both modes tested at 1/2 threads, same-thread parity passed, next iteration executed, source hash unchanged.
- Final-save metric inclusion verified.
- Actual Ryzen integration subsequently exercised the selected 31-worker path and passed.

## Stop/continuation criteria

The fit gate itself is closed. The next finite gate is LT2 Stage A, launched by `tools/run_long_training_lt2_stage_a.sh`: continue the preserved LT1 state for exactly 1,000 additional iterations / 600,000 roots in a separate run directory, with 31 workers, 8 parent Torch threads, vectorized batches and checkpoints every 100 iterations. Stop after Stage A and inspect memory/swap, reservoir saturation, checkpoint size/save time and throughput before any larger LT2 block.

Fail closed on malformed input, nonfinite loss, load error, source hash mismatch, missing report or trainer failure. Do not change strategic settings to obtain speed. If long-run memory/checkpoint overhead becomes dominant, prefer compact reservoir/checkpoint engineering over silently shrinking the 2M reservoir capacity.
