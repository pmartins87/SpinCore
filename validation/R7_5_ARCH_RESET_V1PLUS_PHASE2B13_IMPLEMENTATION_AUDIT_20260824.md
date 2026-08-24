# R7.5 Architecture Reset — Phase2B13 Implementation Audit

Status: **IMPLEMENTED / FROZEN BEFORE OUTPUTS / RYZEN TESTS NOT YET RUN**  
Date: 2026-08-24

## Scope

Audited the Phase2B13 implementation against the frozen Phase2B12 PASS and the Phase2B13 precommit.

Files:

- `validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B12_RESULT_EVIDENCE_20260824.json`
- `validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B13_ROOT_IID64_TARGET_TRAINING_PRECOMMIT_20260824.md`
- `tools/r7_5_arch_reset_v1plus_phase2b13_root_iid64_target_training.py`
- `tools/test_r7_5_arch_reset_v1plus_phase2b13_root_iid64_target_training.py`
- `tools/run_r7_5_arch_reset_v1plus_phase2b13_root_iid64_target_training_ryzen.ps1`
- inherited Phase2B6 training/policy-fit mechanics;
- inherited Phase2B10 explicit-deal root target evaluator;
- inherited Phase2B11 legal conditional-IID private/public deal generator.

Frozen result identities:

- Phase2B6 SHA-256 `33ec6ba89823dae632b7af935def17444379c96a28e59478c0b7c91f1ec3659a`;
- Phase2B12 SHA-256 `dbccadae5805381d0188bef41fb62a72b25b42e03e5564ca88f05d9666e6e182`.

## Findings

### 1. The causal intervention is restricted to the scope actually supported by Phase2B12

Phase2B12 established conditional-IID convergence only at the **initial preflop root**, where the acting player's observable infoset can be held exactly fixed while opponent holes and future board are legally resampled.

Phase2B13 therefore replaces only the one initial-root Advantage sample belonging to the initial root actor. It does not attempt to resample or average arbitrary post-history infosets, where prior opponent actions would condition the hidden-card distribution.

### 2. Equal compute is real, not nominal

Both arms generate and evaluate exactly 64 conditional-IID explicit deals for every logical root using the same deterministic chance seeds and the same fixed traversal RNG seed.

- control: inserts raw target from sample 0;
- candidate: inserts arithmetic mean of all 64 raw targets.

The control deliberately computes and discards samples 1..63. Thus candidate improvement cannot be attributed to a larger root-target traversal budget.

### 3. Reservoir insertion ordering is preserved

A `RootReplacingAdvantageMemory` proxy is installed only during the ordinary logical-root collection call. When the collector reaches the exact initial-root sample, the proxy immediately delegates a replacement `ActionAdvantageSample` to the real reservoir at that exact add position.

This preserves:

- `seen` count;
- reservoir replacement RNG call position;
- ordering relative to all downstream Advantage samples;
- weight and legal-mask semantics.

The proxy requires exactly one matching root replacement per logical root. Zero or multiple replacements are fatal.

### 4. Downstream Advantage and Strategy mechanics remain ordinary

The normal `session.collect_root` still executes once per logical root for all live traversers and the normal Strategy pass. Only the root actor's exact initial-root Advantage sample is substituted.

No downstream Advantage target is averaged or suppressed. No Strategy sample is altered by the target-replacement mechanism. Differences after the intervention can arise causally through subsequent learned behavior, which is intended.

### 5. Phase2B6 behavior intervention is held constant in both arms

Both arms use the same `PreflopContinuationFloorPolicy` from Phase2B6:

- 25% uniform floor only after at least one non-forced preflop event;
- initial root native;
- postflop native;
- heldout AveragePolicy inference floor `0.00`.

Therefore Phase2B13 isolates the root-target estimator on top of the already-supported B6 continuation stabilization rather than changing two causal mechanisms at once.

### 6. Auxiliary chance estimation is detached from live training RNG

The 64 auxiliary explicit deals and their traversal RNG use dedicated deterministic namespaces. Auxiliary workers do not consume the live bundle batch RNG or live reservoir RNG.

After auxiliary rows are sorted back into global-root order, ordinary logical-root collection remains sequential and deterministic inside each arm/seed trajectory.

### 7. Behavior is frozen while a chunk's 64 root estimators are precomputed

Advantage fitting occurs only after the second 64-root chunk of an iteration. Thus behavior models do not change during either chunk before the iteration fit. Precomputing each chunk's auxiliary root targets before its ordinary logical-root collection does not create a stale-model mismatch within the frozen stage mechanics.

### 8. Pilot budget is intentionally smaller than the full x4 confirmation

Per arm/seed:

- 3 iterations;
- 2 × 64 logical-root chunks per iteration;
- 384 logical roots;
- 64 auxiliary root-target traversals per logical root.

Across two arms and two seeds this is 1,536 logical roots and 98,304 auxiliary root-target traversals.

A causal PASS does not certify the architecture. It only authorizes a separately frozen full x4 confirmation.

### 9. Learning mechanics remain inherited

At the end of each iteration, the implementation calls the already-audited zero-root fit helper used by Phase2B6/x4:

- same primary Advantage reset seeds;
- same side-member seeds and batch-seed isolation;
- same Advantage loss and step count;
- same audit and NRMSE gate;
- same reservoir capacity;
- same COMMON/NATIVE AveragePolicy fit protocols.

The root-target estimator is the intended difference.

### 10. Resume semantics are chunk-granular and fail-closed

Each arm/seed writes an independent checkpoint after every 64-logical-root chunk, including:

- arm identity;
- K=64;
- stage index;
- learned behavior states;
- bundle/reservoir state;
- floor-policy telemetry;
- last stage report.

On restart, arm/K/config/execution identity is revalidated. Completed policy artifacts are separately hash-checked before reuse.

### 11. Parallelism uses the Ryzen without changing the experiment

The launcher starts up to two arm/seed trajectories concurrently. During their K64 auxiliary phases each trajectory may use 14 one-thread chance workers, for about 28 independent chance workers total. Random seeds are explicit; worker scheduling does not affect scientific identity.

### 12. Local and final gates are fail-closed

Before causal interpretation, every arm/seed must have:

- exactly 384 logical roots;
- exactly three completed Advantage fits;
- 128 root replacements per iteration;
- 8,192 auxiliary root-target traversals per iteration;
- all Advantage NRMSE gates passing;
- all COMMON/NATIVE policy-fit gates passing.

Final causal comparison uses exact heldout descriptors and a paired equal-group bootstrap. The historical mean `<=0.15` and p95 `<=0.35` gates remain separate.

### 13. Strategic firewall remains intact

Even if the candidate passes the small causal pilot and even if its small-budget heldout happens to satisfy the historical stability thresholds, Phase2B13 does not select H2. A full x4 confirmation is required first, followed by a separately precommitted strategic-strength comparison against the certified stable V1 control.

## Audit conclusion

The implementation matches the frozen Phase2B13 causal question and is suitable for Ryzen execution **subject to** the launcher's real Python compilation/tests, exact prerequisite SHA checks, CMake/MSVC build, x64/ABI checks, explicit-deal round-trip test and runtime local-validity gates.

No Ryzen Phase2B13 output existed at the time of this audit. `PRODUCTION TRAINING = NO`; `READY FOR TABLES = NO`.
