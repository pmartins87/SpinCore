# SpinCore — Long-Training Plan

Status: **TRAINING CLOSED FOR VALIDATED HU CANDIDATE — DEPLOYMENT INTEGRATION**
Date: 2026-09-21

## Final strategic result

ENS8 current behavior at iteration 8100 passed the sealed final holdout under all 11 frozen criteria.

No additional roots are justified for this candidate.

The strategic experiment is closed.

## Frozen artifact identity

Checkpoint:
`a51dbbed71090e45f2c4f5db6297f72eab848990b38650702b436e2aa60ca4bf`

HU ensemble:
`c44b817f75304db352eedc33febbd180e6d038e0580c91be0b9befdbdb08f181`

## Deployment semantics

THREE_HANDED:
- finalized AveragePolicy from iteration 8100;
- unchanged canonical inference semantics.

TRUE_HEADS_UP:
- current ENS8 from iteration 8100;
- eight raw Advantage outputs averaged;
- unchanged lean regret matching.

## Next mechanical gate

Export a compact inference-only bundle and compare its output against source artifacts over old forensic trajectories.

Pass requires:
- zero legal-context mismatches;
- zero argmax mismatches;
- zero exact action-resolution mismatches;
- max probability difference <= 1e-6;
- no strategic EV calculation;
- no holdout reuse.

After parity PASS, proceed to the actual runtime/OpenHoldem integration layer while preserving the frozen bundle hash.
