# R7.5 Architecture Reset — Phase2C2 Implementation Audit

Date: 2026-08-24  
Status: **IMPLEMENTED / FROZEN BEFORE OUTPUTS / RYZEN TESTS NOT YET RUN**

## Trigger

Phase2C1 returned `EXACT_RANGE_REACH_TRANSITION_PROTOTYPE_FEASIBLE` with source-result SHA256 `62ad2352c807a3b046bc84df2cbdf66cc8e0217e3422d01f2bcd9ddeafe7875b`. All 16 structural transition tasks passed: maximum incremental-vs-direct reach error `5.551115123125783e-17`, maximum current-actor scalar error `0.0`, maximum joint-normalizer relative error `2.1541587260971804e-16`, maximum effective-support relative error `3.8543787362290213e-16`, all positive-joint counts matched, raw reach storage was 39,200 bytes, and maximum table-policy evaluations were 4,900.

Phase2C1 therefore permits exactly one separately precommitted bounded structural causal pilot. Phase2C2 is that pilot.

## Finite-route protection

Phase2C2 has only two scientifically admissible completed outcomes:

* causal PASS -> one full x4 hard-stability confirmation of the **same** frozen structural candidate;
* causal FAIL -> select the certified stable V1 fallback and close the V1+ architecture reset.

A local-validity failure is an implementation/audit stop and may be mechanically corrected without changing the scientific algorithm. It does not permit changing K, strata, seeds, gates, target scope, behavior floor, or heldouts.

## Equal-compute causal design

The two arms are:

* `RANGE1_EQUAL_COMPUTE_CONTROL`
* `RANGE64_MEAN_CANDIDATE`

Both arms compute the same 64 root IID targets and use their mean at the initial root. Both arms also compute the same 64 structural posterior continuation targets for one deterministic two-action preflop continuation per logical root. The control inserts one cell selected uniformly from the same 64-cell stratified set and rotated to position zero; the candidate inserts the arithmetic mean. Therefore the control has the full posterior marginal and candidate benefit cannot be attributed to greater auxiliary target compute.

The pilot uses 2 iterations × 64 logical roots per arm/seed = 128 roots per arm/seed. Across 2 arms × 2 training seeds this is 512 logical roots and 65,536 auxiliary target traversals total: 32,768 root-IID64 traversals plus 32,768 structural-continuation traversals.

## Source behavior and causal propagation

Each arm/seed starts with fresh Advantage/Strategy reservoirs and fresh optimizer state. Iteration-1 behavior is bootstrapped from the exact final Phase2B13 `IID64_MEAN_CANDIDATE` four-member Advantage ensemble for that training seed. This prevents the structural sample replacement from being diluted into the already-large historical B13 reservoirs.

After iteration 1 the standard frozen four-member Advantage fit is performed. Iteration 2 therefore uses behavior produced by the intervention, allowing a causal effect to propagate into traversal/Strategy collection and final AveragePolicy fitting.

The 25% uniform behavior floor remains restricted to preflop continuations. Root, postflop, and heldout AveragePolicy inference remain unfloored.

## Deterministic continuation selection

For every logical root the auxiliary worker reconstructs the exact frozen root and searches a positive-support, nonterminal, two-action preflop public path. The action preference is frozen as:

`CHECK_CALL (slot 1) -> aggressive slots 2..9 ascending -> FOLD (slot 0)`.

An action is eligible only if it is currently legal and has positive probability under the frozen behavior (including the preflop continuation floor where applicable). The selected target must remain preflop and contain at least two non-forced preflop events; otherwise the task aborts.

The launcher runs an all-scenario/all-source-seed path preflight before the long pilot.

## Exact reach kernel

At the selected continuation, the final current actor's private cards are fixed and each opponent receives the exact 2,450-entry ordered-hand reach vector. Reach is propagated event-by-event using the already-passed Phase2C1 event-probability update.

For the two opponent vectors `wA` and `wB`, the legal joint posterior is proportional to `wA(hA) * wB(hB)` under exact card-exclusion masking.

The K64 structural proposal is generated without importance or rejection weights:

1. compute the exact collision-adjusted seat-A marginal;
2. divide its CDF into eight equal-probability strata and draw one deterministic seeded random point in each;
3. for each selected seat-A hand compute the exact card-removal-conditioned seat-B CDF;
4. draw eight deterministic seeded stratified seat-B points;
5. obtain exactly 8×8 = 64 legal joint assignments;
6. sample one independent future board per assignment;
7. use one fixed traversal RNG for all 64 continuation target traversals so the arithmetic mean integrates private/public chance rather than traversal RNG.

No rejection, self-normalized importance weighting, clipping, tempering, MCMC, SIR, or K tuning exists in the implementation.

The fair-control patch is installed by the final wrapper entrypoint in the parent, each arm/seed subprocess, and the `ProcessPoolExecutor` chance workers created with Windows `spawn`. A dedicated spawn-context test verifies the stratified-kernel patch, continuation-task patch, entrypoint redirect, and deterministic control-cell index inside a fresh child process before the long pilot can start.

## Exact sample replacement

`MultiReplacingAdvantageMemory` intercepts reservoir `add()` at the exact observation/iteration boundary. It replaces exactly two samples per logical root:

* the initial-root actor sample with the B13 IID64 mean in both arms;
* the selected depth-2 continuation actor sample with control target0 or candidate K64 mean.

The delegate receives exactly one sample for every intercepted `add()`, preserving reservoir seen-count, add order, and RNG call position. Duplicate or missing replacements abort.

The launcher includes a real solver/CFR preflight that constructs one K64 root + K64 structural continuation kernel and then proves exact root+continuation replacement in live collection for both frozen source seeds before the long run begins.

## Checkpoint/resume

Each arm/seed is checkpointed after each 32-root chunk. Resume requires:

* exact execution SHA;
* Phase2C2 checkpoint schema;
* exact arm/K/iteration/chunk contract;
* exact frozen stage config;
* exact source B13 behavior-checkpoint SHA;
* matching completed iteration/global root.

Untracked local artifacts do not dirty the tracked worktree gate. Existing valid Phase2C2 checkpoints are resumed; they must not be deleted on interruption.

## Local and causal gates

Local validity requires all expected root/continuation replacements and auxiliary traversal counts, all Advantage weighted NRMSE gates <=0.75, and all final AveragePolicy fit weighted mean TV gates <=0.12.

The frozen primary causal gates are the ones in the precommit: COMMON pooled materiality >=0.020 absolute or >=10% relative, strictly positive bootstrap lower bound, both heldouts improve in mean, p95 nondegradation <=0.020, continuation-2plus improves in both heldouts, root nondegradation <=0.020, and NATIVE noncontradiction.

Hard stability (`mean <=0.15`, `p95 <=0.35` on both COMMON heldouts) is recorded but is not required to pass this small pilot. A causal PASS only permits the later full x4 confirmation.

## Files

* `validation/R7_5_ARCH_RESET_V1PLUS_PHASE2C1_RESULT_EVIDENCE_20260824.json`
* `validation/R7_5_ARCH_RESET_V1PLUS_PHASE2C2_RANGE_REACH_TARGET_KERNEL_CAUSAL_PILOT_PRECOMMIT_20260824.md`
* `tools/r7_5_arch_reset_v1plus_phase2c2_range_reach_target_kernel_causal_pilot.py`
* `tools/test_r7_5_arch_reset_v1plus_phase2c2_range_reach_target_kernel_causal_pilot.py`
* `tools/r7_5_arch_reset_v1plus_phase2c2_range_reach_target_kernel_causal_pilot_controlfair_v2.py`
* `tools/test_r7_5_arch_reset_v1plus_phase2c2_controlfair.py`
* `tools/test_r7_5_arch_reset_v1plus_phase2c2_controlfair_spawn.py`
* `tools/test_r7_5_arch_reset_v1plus_phase2c2_live_replacement.py`
* `tools/run_r7_5_arch_reset_v1plus_phase2c2_range_reach_target_kernel_causal_pilot_controlfair_ryzen.ps1`

## Real-execution status

No real Windows/Ryzen Phase2C2 output has been observed when this audit was written. The launcher is responsible for the frozen Python check, synthetic tests, explicit Windows-spawn fair-control propagation test, exact source-result/checkpoint hashes, fresh VS2022 x64 solver build, PE/ABI/SPNNIV3/explicit-deal checks, explicit-deal round-trip regression, all-scenario depth-2 path preflight, live structural K64 kernel preflight, and live two-sample replacement preflight before the pilot is allowed to start.

No PASS is claimed for any of those real-machine checks until the user runs the frozen launcher.
