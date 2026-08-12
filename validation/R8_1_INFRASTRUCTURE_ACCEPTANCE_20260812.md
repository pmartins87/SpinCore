# R8.1 — integrated production infrastructure acceptance — 2026-08-12

`R8.1 INFRASTRUCTURE GATE = PASS`

`R8 OFFICIAL TRAINING AUTHORIZED = NO`

`READY FOR TABLES = NO`

This record closes the finite R8.1 infrastructure gate defined before production training. It does not override the independent R7.4 final prerequisite, does not manufacture the still-missing exact GGPoker R8.0 production profile, and does not close the deferred R7.3 exact-reproducibility release debt.

## Accepted contracts

The accepted production infrastructure is the composition of:

```text
SPINCORE_R8_PRODUCTION_PROFILE_V3
SPINCORE_R8_INDEPENDENT_STREAM_SCHEDULER_V2
SPINCORE_R8_CENTRAL_ALGORITHM_R_V2
SPINCORE_R8_SCHEDULER_DURABLE_CHECKPOINT_V1
SPINCORE_R8_PRODUCTION_TRANSACTION_V1
SPINCORE_R8_PRODUCTION_TRANSACTION_POINTER_V1
```

### 1. Exact production identity / evidence binding

A profile-dependent production fact cannot be justified merely by pointing at a generic official dynamic page. `SPINCORE_R8_PRODUCTION_PROFILE_V3` requires selected-state evidence bound to the exact table-size, buy-in and multiplier state for the fields that evidence claims to prove.

Regression:

```text
workflow run 31638697150 = PASS
```

### 2. Serial RNG stream scheduler

Concurrency is allowed only across independently identified `(profile, domain, algorithm_seed)` streams. One stream receives at most one whole-iteration lease at a time. The scheduler does not derive per-root RNGs and cannot authorize intra-stream root parallelism that would alter the selected persistent-RNG algorithm.

A completed iteration advances only after a durable receipt names checkpoint locator, SHA-256, byte size and parent checkpoint SHA-256. Crash recovery clears an unverifiable active lease and retries the same logical iteration rather than advancing it.

### 3. Exact Algorithm-R stream identity

`SPINCORE_R8_CENTRAL_ALGORITHM_R_V2` is bound to:

```text
profile_id
× domain
× algorithm_seed
× roots_per_iteration
```

It rejects cross-seed mixing, duplicate/stale roots, non-contiguous completion gaps at drain, root/iteration disagreement, and samples whose own iteration differs from their root batch. Worker wall-clock completion order cannot change the logical ascending-global-root insertion order.

Regression:

```text
workflow run 31639134952 = PASS
```

### 4. Durable scheduler state

Scheduler state is persisted with same-directory temporary write, file fsync, atomic replacement and directory fsync where supported. Loading is schema- and hash-checked and fails closed on corrupted/truncated state.

Regression:

```text
workflow run 31639331611 = PASS
```

### 5. Integrated production transaction

The final missing R8.1 risk was cross-component skew: individually valid model/RNG, scheduler and Algorithm-R checkpoints could have represented different moments or different streams.

`SPINCORE_R8_PRODUCTION_TRANSACTION_V1` resolves this with immutable generation directories and an atomically replaced `CURRENT.json` pointer. A generation is publishable only when all component bytes are durable and their semantics agree.

The transaction verifies:

- stream checkpoint schema is `SPINCORE_R7_CHECKPOINT_V2`;
- stream domain and algorithm seed equal the transaction identity;
- stream `progress.iteration` equals the completed iteration;
- scheduler checkpoint/state schemas are the accepted R8.1 schemas;
- exactly one scheduler stream matches profile/domain/seed;
- no active lease exists at publication;
- scheduler `next_iteration = completed_iteration + 1`;
- scheduler last checkpoint SHA-256 and byte size equal the exact stream checkpoint bytes;
- Algorithm-R profile/domain/seed/roots-per-iteration equal the transaction identity;
- `next_global_root = committed_roots = completed_iteration × roots_per_iteration`;
- no pending Algorithm-R root gaps exist;
- every component and pointer remains `ready_for_tables = false`.

The loader rechecks component SHA-256/size, generation identity, manifest pointer and semantic consistency. Tests additionally prove fail-closed behavior for component tampering, cross-seed component mixing, wrong scheduler stream SHA, wrong Algorithm-R root position and missing/incomplete components.

Authoritative integrated regression:

```text
workflow run: 31650375049
head: 0bcede8c07c635527adb6b69e5aef512461790e7
conclusion: PASS
C++ regression: PASS
Python tooling syntax: PASS
Python regression: PASS
```

## R8.1 conclusion

The finite infrastructure questions precommitted for R8.1 are now answered with fail-closed implementations and regression evidence. Therefore:

```text
R8.1 deterministic production infrastructure = PASS
```

This is an infrastructure PASS only. It is **not** evidence that a production policy has been trained or that a production economic profile is complete.

The following remain blocking before official R8 training can begin:

```text
R7.4 final gate = must PASS
R8.0 exact GGPoker production profile = must PASS
```

The following remains blocking before final table homologation:

```text
R7.3 exact historical reproducibility debt = OPEN / NOT PASS
```

No R8.2 calibration, R8.3 HU production training, R8.4 3H production training or R8.5 production-policy freeze is authorized by this document alone.
