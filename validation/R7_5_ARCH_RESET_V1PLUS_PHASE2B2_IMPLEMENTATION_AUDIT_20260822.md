# R7.5 Architecture Reset — V1+ Phase2B2 Implementation Audit

Status: **IMPLEMENTATION AUDITED BEFORE OUTPUTS**  
Date: 2026-08-22

The Phase2B2 implementation was reviewed against the frozen precommit before any Phase2B2 scientific output exists.

## Identity and source controls

- Requires the exact local Phase2B1 JSON and verifies its SHA-256 against `R7_5_ARCH_RESET_V1PLUS_PHASE2B1_RESULT_EVIDENCE_20260822.json`.
- Requires Phase2B1 status `PHASE2B1_K4_SCREEN_FAIL_NO_GENERIC_K4_TRAINING_PILOT`, classification `CHANCE_DOMINANT`, and no generic K4 pilot authorization.
- Uses the two frozen Phase2A H2 / THREE_HANDED behavior ensembles from source execution SHA `4bfa55d69029cd69536fa6dbfcadd162719cb887`.
- Requires exactly four behavior members per source seed and the completed 768-root / stage-12 Phase2A checkpoints.
- Uses exactly the 15 Phase2B1 collision groups and their 16 stored deck seeds. No new collision search is performed.

## Causal controls

- Every stored deck seed is recreated and its exact root SPNNIV3 SHA-256, actor, legal set and legal mask are checked before target extraction.
- Both frozen behavior ensembles therefore see the same deal/chance realization for each paired comparison.
- `COMMON_TRAVERSAL_RNG` uses the same deterministic traversal RNG seed for both behaviors.
- `INDEPENDENT_TRAVERSAL_RNG` uses separate deterministic RNG namespaces and is the primary decision arm, preventing a favorable common-random-number coupling from being mistaken for a chance-support effect.
- Target extraction requires exactly one root Advantage sample matching the exact root observation hash and legal mask.

## Learned-state immutability

The diagnostic:

- does not call any optimizer;
- does not fit or reset a network;
- does not insert into any training reservoir;
- uses fresh in-memory sinks for diagnostic Advantage samples;
- does not modify or resave source checkpoints.

Fresh solver traversal is the only stateful computation and is required to measure target generation under paired chance support.

## Statistical implementation

For every `(scenario, arm)` there are 16 paired target replicates. For K=1/2/4/8/16, raw ten-slot targets are averaged separately for the two behavior ensembles over identical non-overlapping deal blocks, then compared using target MAD, legal sign disagreement, regret-matching policy TV and dominant-action mismatch.

Pooled pair counts per arm are 240/120/60/30/15 for K=1/2/4/8/16.

The primary shared-support gate is frozen to `INDEPENDENT_TRAVERSAL_RNG` K1 and requires simultaneously:

- at least 0.10 absolute improvement versus Phase2B1 chance-only K1 TV;
- at least 30% relative improvement;
- absolute cross-behavior K1 TV <= 0.35;
- COMMON_TRAVERSAL_RNG K1 not more than 0.05 worse than the independent arm.

If independent same-chance K1 remains >=80% of the Phase2B1 chance reference, behavior feedback is classified as remaining dominant on common chance support.

## Compute audit

Frozen work: 15 scenarios × 16 deals × 2 source behaviors × 2 RNG arms = 960 root Advantage traversals.

Launcher uses up to 12 Windows worker processes with one Torch/OMP/MKL/OpenBLAS thread each. Worker completion order cannot affect metrics because rows are sorted before aggregation.

## Governance conclusion

Implementation matches the frozen Phase2B2 causal question. Phase2B2 cannot authorize production or table use. Only a shared-support gate PASS can authorize precommitting one small shared-chance-support training pilot, and that later pilot must retain independent chance-block validation.
