# R7.5 Architecture Reset — Phase2B5 Implementation Audit

Status: **IMPLEMENTATION AUDITED BEFORE OUTPUTS**  
Date: 2026-08-22

The Phase2B5 implementation was reviewed against the frozen precommit before any Phase2B5 scientific output existed.

## Audited invariants

- `PREFLOP_NATIVE_POSTFLOP_COMMON` retains native source behavior only on preflop continuation calls and commonizes every postflop policy call.
- `DEPTH_COMMON_GE_1` commonizes every downstream preflop policy call because the first child after a fixed root action has non-forced preflop-event delta >= 1.
- Depth is derived only from authoritative SPNNIV3 public history. The parser requires exact wire size `120 + 20 * history_count` and counts only `street==0 && forced==0` events.
- Root sigma is the same per-deal arithmetic mean used in Phase2B4, so the Phase2B3 root-baseline effect remains controlled out.
- Uniform-floor arms mix only legal-action probability mass and renormalize; illegal action slots remain zero.
- Uniform-floor smoothing is applied only in preflop continuation and only after the already-frozen source behavior (including its existing uncertainty damping).
- Flop/turn/river continuation is commonized identically in every arm.
- Both source sides use the same stored deal support and the same independent traversal-RNG namespaces as the prior diagnostics.
- `PREFLOP_NATIVE_POSTFLOP_COMMON` must reproduce Phase2B4 `COMMON_FROM_FLOP` TV `0.32010786853721923` within `1e-12` or abort.
- `DEPTH_COMMON_GE_1` must reproduce Phase2B4 `COMMON_FROM_PREFLOP` TV `0.060271017892879135` within `1e-12` or abort.
- Pilot thresholds are encoded exactly as frozen: abs reduction >=0.08, relative >=25%, residual <=0.24, >=12/15 scenarios improve, max scenario degradation <=0.05, dominant-action mismatch increase <=0.02.
- Only floors 0.10 or 0.25 may produce `MILD_PREFLOP_DAMPING_CANDIDATE` and authorize freezing a small pilot. 0.50 or larger can never authorize a pilot in this screen.
- The launcher ignores untracked files but stops on any tracked-worktree modification.
- No optimizer step, model fit, reservoir insertion, checkpoint mutation, or architecture selection is present in the diagnostic path.

## Compute

12 arms x 15 scenarios x 16 stored deals x 2 source behaviors = 5760 root action-value reconstructions, with at most 12 worker processes and one Torch/OMP/MKL thread per worker.

`READY FOR TABLES = NO`.
