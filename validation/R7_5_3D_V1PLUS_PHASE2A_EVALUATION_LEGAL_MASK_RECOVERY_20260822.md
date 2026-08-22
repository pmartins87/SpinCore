# R7.5.3D — Phase 2A evaluation legal-mask recovery

Date: 2026-08-22
Status: MECHANICAL_EVALUATION_RECOVERY_FROZEN_BEFORE_RESULT
Source execution SHA: `4bfa55d69029cd69536fa6dbfcadd162719cb887`
READY FOR TABLES: NO
Production training authorized: NO

## Failure boundary

The Phase 2A source execution completed both H2/3H x4 training-seed trajectories, exact Strategy-stream replay, all six AveragePolicy fits per training seed, and both seed-level local Advantage gates. The run then failed only in the parent heldout evaluation while constructing a PyTorch legal-action tensor.

The frozen heldout artifact stores `legal_slots` as a variable-length tuple of universal action indices. The Phase 2A evaluator passed those tuples directly to `collate_v3_observations`, whose `_legal_tensor` contract requires a fixed-width ten-element boolean mask. This produced `ValueError: expected sequence of length 4 at dim 1 (got 3)` before any cross-seed Phase 2A metric or final result JSON was produced.

The authoritative final V3 policy evaluator already performs the correct conversion: `legal_mask(row)` converts a variable-length universal legal set into the ten-slot mask before collation.

## Permitted recovery

Recovery is evaluation-only and MUST consume the already-completed artifacts from source execution SHA `4bfa55d69029cd69536fa6dbfcadd162719cb887`.

Permitted change:

- convert every heldout `legal_slots` tuple to the canonical ten-slot universal mask with `spincore.r7_5_action_cfr.legal_mask` before calling `collate_v3_observations`;
- validate the resulting probability vector against the original legal set with `validate_policy`, matching the established final-policy evaluation path;
- recompute only heldout probabilities, cross-seed TV/bootstrap statistics, Phase 2A decision status, and final result JSON.

Forbidden:

- no new traversal;
- no new roots or deck seeds;
- no Advantage retraining;
- no Strategy reservoir replay change;
- no AveragePolicy refit;
- no changed learner seed, optimizer step, capacity arm, threshold, heldout state, or selection rule;
- no production authorization.

## Artifact identity

The recovery evaluator must reject either seed result unless it reports `SEED_COMPLETE` under the exact source execution SHA. It must require all twelve policy artifacts (2 training seeds × 3 capacity arms × 2 learner modes) to exist before evaluation.

The recovered result remains scientifically attributed to the source execution SHA; the recovery commit identifies only the mechanical evaluator correction.

## Interpretation

This correction cannot improve or worsen a policy because it does not modify a model. It only converts the already-frozen heldout representation of legal action indices into the ten-slot mask required by the neural batch collation API. Any Phase 2A capacity-effect conclusion must come from the recovered heldout metrics, not from the failure itself.
