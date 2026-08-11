# R7.3 candidate checkpoint/resume readiness — 2026-08-10

`READY FOR TABLES = NO`. This note does not promote any R7.3 candidate and does not change any frozen gate.

## Why this work is needed before a 640 acceptance run

The recovered authoritative checkpoint schema is `SPINCORE_R7_CHECKPOINT_V2`. It correctly preserves the Generation-2 `DomainBundle` state used through R7.2:

- primary AdvantageNet;
- AveragePolicyNet;
- both optimizer states;
- Advantage and Strategy reservoirs;
- persistent `bundle.batch_rng`;
- PyTorch RNG state;
- counters;
- exact mid-iteration progress;
- arbitrary `extra` metadata.

That is sufficient for the recovered single-Advantage-network behavior and is covered by the existing exact continuous-vs-stop/restore/continue regression.

The active R7.3 candidates add behavior state that is **not** part of a plain `DomainBundle`:

- size8 policy mixture: seven side Advantage models in addition to the authoritative primary model;
- temporal blend: the current ensemble plus the previous iteration's ensemble and blend weight;
- uncertainty damping: the current ensemble plus adaptive-damping parameters.

Therefore the existing R7.2 checkpoint test cannot simply be cited as proof for a promoted ensemble candidate. A candidate must preserve and restore this additional behavior state exactly before it can be acceptance-scaled.

## Implemented preparation

`python/spincore/r7_candidate_checkpoint.py` adds `SPINCORE_R7_CANDIDATE_BEHAVIOR_V1` as an **extra-payload schema**, deliberately leaving the authoritative base checkpoint schema unchanged.

The contract is:

1. `current_models[0]` must be the exact authoritative `bundle.advantage` object;
2. member zero is also fingerprinted by an exact cloned state dict in the candidate payload;
3. only side current members are reconstructed as additional networks;
4. temporal previous-generation members are serialized independently;
5. wrapper parameters and fit generation are stored explicitly;
6. restore fails closed if the primary model loaded by the base checkpoint does not exactly match the candidate payload;
7. restored `current_models[0]` reuses the already-restored authoritative primary object, preventing an accidental duplicate member-zero network.

The candidate payload is stored through the existing checkpoint `extra` field. This keeps R7.2 state compatibility separate from R7.3 experimental behavior semantics.

## Added regression coverage

`python_tests/test_r7_candidate_checkpoint.py` covers:

- rejection when current member zero is not the authoritative primary object;
- size8-style current-ensemble roundtrip;
- temporal current + previous ensemble roundtrip;
- snapshot isolation from later model mutation;
- fail-closed primary-state mismatch;
- survival of the candidate payload through the real `save_checkpoint` / `load_checkpoint` `extra` path.

These are serialization/readiness tests only. They are **not** a substitute for the required physical recertification.

## Remaining mandatory recertification after winner selection

Once a fit-valid five-iteration candidate is selected, the exact winning behavior semantics must be frozen and a physical deterministic comparison must run:

```text
continuous execution
vs
execute -> checkpoint -> fresh restore -> continue
```

The comparison must include, at minimum:

- counters;
- Advantage and Strategy reservoir contents/order;
- `bundle.batch_rng` continuation;
- primary Advantage model;
- every side Advantage model;
- previous ensemble when the winner is temporal;
- wrapper parameters/generation;
- AveragePolicy model;
- final cross-seed metrics.

Any mismatch blocks 640 acceptance scaling.

## Current state

The serialization helper and tests are committed on `main`. The normal main regression is the gate for this preparation. Physical candidate recertification remains intentionally blocked until the five-iteration winner is known, so we do not freeze or certify semantics that may be discarded.
