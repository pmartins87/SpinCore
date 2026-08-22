# R7.5.3D — Phase 2A parallel AveragePolicy fit execution freeze

Date: 2026-08-21
Status: FROZEN_BEFORE_PHASE2A_OUTPUTS
READY FOR TABLES: NO
Production training authorized: NO

## Purpose

Improve utilization of the Ryzen 9 9950X during the mechanically independent final AveragePolicy fits of the already-frozen Phase 2A Strategy-memory capacity ablation, without changing any scientific dimension of the experiment.

The upstream traversal/Advantage trajectory remains exactly as precommitted. This document freezes execution scheduling only.

## Frozen scheduling

Upstream collection:

- two independent training-seed processes may run concurrently;
- within each training seed, the x4 trajectory remains strictly sequential in canonical root/chunk order;
- `SPINCORE_TORCH_THREADS=2` remains unchanged;
- no root-level parallelism is introduced;
- deck seed, scenario order, Advantage RNG, Strategy stream, model architecture, optimizer budgets and reservoir semantics are unchanged.

Final AveragePolicy fitting after a seed's Strategy stream has been completed and replayed:

- the three passive capacity arms `S100K_CONTROL`, `S400K`, and `S800K` may run in three isolated child processes concurrently;
- each child process receives an exact serialized state of only its already-built Strategy reservoir plus the authoritative final `bundle.batch_rng` state;
- each child process uses exactly two PyTorch intra-op threads;
- `COMMON_LEARNER` and `NATIVE_LEARNER` remain sequential inside each capacity process so one reservoir is loaded once and each learner preserves the frozen RNG semantics;
- with both training seeds fitting simultaneously, the maximum intended heavy fit concurrency is six arm processes × two PyTorch threads = twelve intra-op threads, plus lightweight parent-process overhead;
- no arm process can mutate the authoritative traversal bundle or another arm's reservoir.

## Scientific identity preservation

Parallel scheduling is permitted only because, after Strategy-stream generation, the six policy fits per training seed are read-only with respect to traversal and mutually independent conditional on their frozen reservoir and learner RNG state.

For every capacity arm the worker must preserve:

- exact retained Strategy items and weights;
- exact `seen`, capacity and reservoir RNG state;
- H2 representation and model architecture;
- policy optimizer: Adam;
- learning rate: 0.001;
- batch size: 256;
- optimizer steps: 16,384;
- COMMON learner initialization seed: `0x13579BDF`;
- COMMON batch RNG seed: `0x2468ACE013579BDF`;
- NATIVE learner initialization and batch RNG semantics already frozen in Phase 2A;
- authoritative policy-audit seed: `training_seed ^ 0x71A5BEEF`;
- unchanged local policy-fit TV gate and cross-seed hard stability gates.

No scientific comparison may depend on wall-clock order of independent arm completion.

## Resume and evidence

Temporary fit contexts are atomic files and are preserved on worker failure/interruption. Completed policy artifacts are independently resumable by `(training_seed, learner_mode, arm, capacity, authoritative audit seed)` identity.

After all six policy artifacts for a training seed are verified complete, temporary large fit contexts may be deleted because they are execution-only duplicates of the already-preserved Strategy reservoirs/streams.

The frozen Ryzen evidence runner must hash this freeze, the parallel fit worker, runtime guard, base Phase 2A runner, deterministic tests, and final output directory.

## Resource rationale

The previous x16 run underutilized a 16-core/32-thread Ryzen because each independent cell was intentionally constrained to two PyTorch threads and long sequential segments existed. Phase 2A has only two upstream seed trajectories, so root collection cannot saturate the CPU without introducing a new traversal algorithm.

The final policy-fit stage has safe coarse-grained parallelism. Three capacity-arm processes per seed are therefore the maximum frozen fit parallelism for this experiment. This targets materially better CPU use while leaving headroom on a 64 GB host and avoiding twelve simultaneous duplicate reservoir processes (COMMON and NATIVE are not split into separate processes).

## Guardrails

- no root-level parallelism;
- no increased PyTorch thread count per scientific fit;
- no seed shopping;
- no threshold relaxation;
- no capacity-arm change;
- no policy-step reduction or increase;
- no representation or action-abstraction change;
- no H2/H3 winner declaration;
- no production authorization;
- stability remains an eligibility gate, not a strategic-strength metric.
