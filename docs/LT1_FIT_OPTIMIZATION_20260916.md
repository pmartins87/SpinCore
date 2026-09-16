# LT1 neural-fit optimization — 2026-09-16

Status: candidate implemented and locally validated; actual Ryzen benchmark pending.

## Evidence and decision

The full LT1 report attributes 71.08% of measured time to Advantage fitting. Current batching decodes 1,024 SPNNIV1 payloads to Python dataclasses/lists every optimizer step. That is a concrete avoidable cost, but the fraction of fit time it consumes was not previously measured. The prior thread benchmark used batch 256 and 30k reservoirs, whereas LT1 uses batch 1024 and 2M capacity. Its 8-thread selection is not evidence of optimality at LT1 scale.

Legacy preflight performed against the supplied Tentativas anteriores de SpinGo.zip: deepspin/buffers.py (array-backed sample_batch), trainer.py (array-to-tensor fit), and networks.py. Preserve the array batching principle, not its different 292-feature MLP or sampling-with-replacement semantics. Current Python random.sample, LT1 wire bytes, reservoirs, optimizer, weighted loss, network reset, 32-token padded GRU history, ten-slot external legal mask, sampler, actions and utility remain unchanged.

## Implemented

- Optional vectorized batch construction: `python/spincore_nn/lean_batch.py`.
- Reference path remains default; historical V2 paths unchanged.
- Fit timings distinguish sample selection, batch assembly and optimizer work.
- Trainer accepts `--batch-mode reference|vectorized`, explicitly applies the requested parent thread count and records it.
- Final report includes the finalized checkpoint save metric and measured wall scope. Terminal prints a compact final summary; complete history stays in report.json.
- No concurrent-domain fit introduced: it adds scheduling/RNG complexity before measurement of fit subphases identifies the need.

## One finite physical benchmark

Run `bash tools/benchmark_lean_lt1_fit.sh` in the existing WSL checkout. Optional first argument is an explicit finalized checkpoint path. Default source is the known LT1 run 20260915_181249.

The source is loaded read-only, SHA256 checked before/after, never passed to save_checkpoint. No reservoir copy is written and no original artifact is overwritten. Fitting operates on in-memory network state and uses the real loaded reservoirs. Sampling RNG and reset seeds are restored between cases. No steps, batch sizes or reservoir capacities are reduced.

Compare 1/2/4/8/16 threads, reference and vectorized batches: 10 finite cases with 2 warm-up steps then the checkpoint-configured full fit steps per domain. Exact tensor parity and same-thread final model hash/RNG equality are mandatory. Thread changes use the same intended training algorithm, but cross-thread bitwise equivalence is not claimed. Select only if measured fit gain is at least 5%; otherwise keep reference/8 and stop. This is a throughput screen, not a formal significance or strategy gate.

Then restore LT1 model/RNG/counters and execute exactly one resumed full iteration with the selected mode and existing root-worker profile. This disposable integration iteration is not saved or promoted to LT2. Emit report.json and stop. A 20-minute external timeout bounds failure/stall; no repeated search or automatic long training. User sends the report for the next decision. Original LT1 stays the source for the real extension.

## Local validation

- 13 targeted pytest cases passed: wire/tensor parity for batches 1/37/1024, exact model and Adam-state parity for Advantage/Policy updates, malformed payload rejection, historical model and training tests.
- Solver compiled in Release (existing compiler warnings remain).
- End-to-end tiny finalized checkpoint loaded by benchmark, both modes tested at 1/2 threads, same-thread parity passed, next iteration executed, source hash unchanged.
- Final-save metric inclusion verified.
- Local multiprocessing sockets are unavailable in this execution environment, so integration smoke used one worker. Existing parallel-root implementation is unchanged; actual selected-worker integration is exercised by the Ryzen command.
- No actual Ryzen speedup is claimed. Full LT1 checkpoint remains on user's machine.

## Stop/continuation criteria

Fail closed on malformed input, tensor/model/RNG mismatch, nonfinite fit loss, load error, source hash change or resumed root-count mismatch. Do not change strategic settings to obtain speed. If screen shows less than 5% gain, preserve baseline and use the measured phase profile to decide whether further engineering is justified. Do not rerun a matrix merely to obtain a winner. After the physical result, continue LT1 into a separate LT2 run directory; retain the original LT1.
