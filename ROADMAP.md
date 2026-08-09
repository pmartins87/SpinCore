# SpinCore finite roadmap — recovered canonical continuation

The final endpoint is **ready to start using at the tables**. `READY FOR TABLES = NO` until every required gate passes.

- R0 Foundation / canonical repository — PASS
- R1 Complete poker engine — PASS
- R2 Canonical infoset + neural encoder — PASS
- R3 Tournament continuation value (`ICM_EXACT_V1`, explicit payout) — PASS
- R4 Neural infrastructure — PASS
- R5 CFR correctness oracle — PASS
- R6 Deep CFR integration on authoritative `SpinTraversalState` — **PASS RECERTIFIED 2026-08-09**
- R7 Pilot / performance / statistical stability — IN PROGRESS
  - R7.0 instrumentation / approximation metrics — **PASS RECERTIFIED**
  - R7.1 batching + native own-reach frontier — **PASS RECERTIFIED**
  - R7.2 distributed reservoir audit + LCFR weighting + mid-iteration resume + fresh-process training workers — **PASS RECERTIFIED**
  - R7.3 multi-seed stability — **FAIL / IN PROGRESS**
  - R7.4 larger HU + 3H pilot — NEXT after the R7.3 diagnostic checkpoint
- R8 Production training — TODO
- R9 Strategic audit — TODO
- R10 OpenHoldem runtime — TODO
- R11 Safe exploitation — TODO
- R12 Operational homologation — TODO

## Frozen R7.3 gates

- Advantage weighted normalized RMSE <= 0.75
- Average-policy weighted mean TV <= 0.12
- Cross-seed mean TV <= 0.15
- Cross-seed p95 TV <= 0.35

Historical last HU checkpoint before chat-runtime loss: **640 roots/seed, mean TV 0.3714, p95 TV 0.6878**. Therefore R7.3 remains FAIL.

## Recovery invariants

- No frozen gate may be relaxed to accelerate recovery.
- `TRUE_HEADS_UP` and `THREE_HANDED` remain separate strategic domains for the entire hand.
- Production utility is exact ICM continuation delta with payout explicitly supplied per run; chip delta is diagnostics only.
- Ambiguous simultaneous elimination under unequal payouts fails closed.
- Every material recovery/roadmap advance must be persisted to GitHub `main`; no recovery branch.
