# R7.5 Architecture Reset — V1+ Phase2B1 Implementation Audit

Status: **READY FOR RYZEN DIAGNOSTIC EXECUTION**  
Date: 2026-08-22

## Audited implementation

Files:

- `validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B1_TARGET_VARIANCE_PRECOMMIT_20260822.md`
- `tools/r7_5_arch_reset_v1plus_phase2b1_target_variance.py`
- `tools/test_r7_5_arch_reset_v1plus_phase2b1_target_variance.py`
- `tools/run_r7_5_arch_reset_v1plus_phase2b1_target_variance_ryzen.ps1`

## Mechanical identity

The diagnostic consumes only the completed Phase2A H2/THREE_HANDED behavior checkpoints from execution SHA `4bfa55d69029cd69536fa6dbfcadd162719cb887` and requires both frozen training seeds. It validates the authoritative Phase-2 representation/action contract before traversal.

The Phase2B0 failed-screen evidence is a hard prerequisite. The rejected algebra candidate is not trained and is not reused as behavior.

## Work count

Frozen diagnostic structure:

- 15 THREE_HANDED scenario collision groups;
- 2 frozen source behavior ensembles;
- 3 variance arms per scenario/behavior;
- 16 target replicates per arm;
- total root-actor Advantage target traversals: `15 * 2 * 3 * 16 = 1440`;
- K pair comparisons per task: `8 + 4 + 2 + 1 = 15`;
- tasks: `15 * 2 * 3 = 90`;
- expected total K pair metric rows: `90 * 15 = 1350`.

No strategy traversal is performed. Each target traversal runs one root actor as traverser only.

## Causal isolation

`TRAVERSAL_ONLY` fixes the exact deck seed and changes only collector traversal RNG.

`CHANCE_ONLY` uses 16 distinct deck seeds that all reproduce the identical exact root SPNNIV3 observation, actor and legal action set, while using one fixed traversal RNG seed.

`COMBINED` varies both according to the frozen deterministic namespaces.

The target captured for each run must match the exact root observation and legal mask exactly once. Missing or duplicate root-target identity is fatal.

## No-training guard

The diagnostic collector writes Advantage samples only into a local `_Sink`; no `UniformReservoir` is passed to the target diagnostic. No call to `train_step`, `train_advantage`, `train_average_policy`, optimizer `.step()`, or model fitting exists in the Phase2B1 tool.

Fresh solver traversal is intentional and is the measured process; model parameters remain read-only.

## Parallelism

Ryzen execution uses up to 12 independent worker processes. Every process is limited to one Torch/OMP/MKL/OpenBLAS thread. Statistical aggregation is independent of worker completion order because task outputs are sorted and K partitions are deterministic.

The launcher builds a clean AMD64 ABI2 solver and verifies the SPNNIV3 C API before the diagnostic.

## Fail-closed behavior

The run stops rather than weakening the experiment if:

- a frozen checkpoint identity fails;
- Phase2B0 routing evidence is absent or does not say FAIL;
- any of the 15 scenarios cannot find a 16-deck exact-root observation collision within 50,000 candidates;
- a replicate does not reproduce the frozen root observation/actor/legal identity;
- a root target cannot be extracted exactly once;
- runtime/model/source contracts drift.

## Governance

Phase2B1 remains diagnostic. Even a K4 screen PASS only allows precommitting one small causal pilot. It does not reopen R7.5.3, select H2/H3, authorize R7.5.4/R8, authorize production, or make the system ready for tables.
