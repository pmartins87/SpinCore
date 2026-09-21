# SpinCore Current Work

Date: 2026-09-21
Status: **NATIVE C++ INFERENCE PASS — HIDDEN-FILLER INVARIANCE PASS — EXACT PUBLIC-TRANSCRIPT REBUILD NEXT**

## Frozen deployment

Python hybrid deployment:
`87e46b40cb43bb89cb46bf3b760bbac5c8282491fd3d6da73d1bbe28329b278c`

Native C++ deployment:
`2b79ab7ff746a9c1c3dd73dbc0a1d6884a471813cf34b9cb126790c4c4cbb123`

No strategic changes are allowed.

## Hidden-card reconstruction result

PASS over:
- 6,000 current states;
- 36,000 alternate hidden-card completions;
- 145,734 legal exact-action resolution checks;
- both domains;
- all four streets.

Changing opponent holes and unrevealed future board cards caused zero differences in:
- actor/domain;
- SPNNIV1;
- SPNNIV2;
- lean legal actions;
- exact action resolution.

## Runtime architecture consequence

The OpenHoldem bridge may safely generate deterministic fillers for currently hidden cards.

However, a persistent filler board cannot simply survive a future street reveal if the real card differs from the filler.

Therefore the bridge will reconstruct the authoritative state from the hand start at each Hero decision:
- original tournament scenario;
- Hero hole cards;
- currently visible board;
- deterministic fillers for remaining hidden cards;
- exact public voluntary action transcript.

## Active gate

Prove that replaying the exact public action transcript through the authoritative `apply_exact` solver API reproduces the live canonical state exactly at Hero decisions.

No inference, EV, training or holdout reuse.

## Immediate action

```bash
bash tools/run_lt2_runtime_exact_transcript_rebuild.sh
```

Wait for `LT2_RUNTIME_EXACT_TRANSCRIPT_REBUILD_PASS`, then send
`SpinCore_LT2_runtime_exact_transcript_rebuild.json`.
