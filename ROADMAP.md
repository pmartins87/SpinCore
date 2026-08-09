# SpinCore finite roadmap — canonical recovery generation 2

Final endpoint: **ready to start using at the tables**. `READY FOR TABLES = NO` until all gates pass.

- R0 Foundation / canonical repository — **PASS REBUILT**
- R1 Complete poker engine — **PASS REBUILT**
- R2 Canonical infoset + neural encoder — **PASS REBUILT**
- R3 Tournament continuation value (`ICM_EXACT_V1`, explicit payout) — **PASS REBUILT**
- R4 Neural infrastructure — **PASS REBUILT**
- R5 CFR correctness oracle — **PASS REBUILT**
- R6 Deep CFR integration on authoritative `SpinTraversalState` — **PASS REBUILT**
- R7 Pilot / performance / statistical stability — IN PROGRESS
  - R7.0 approximation metrics / full-reservoir audit — **PASS REBUILT**
  - R7.1 native own-reach frontier — **PASS REBUILT**
  - R7.2 LCFR weighting / exact checkpoint+resume / fresh-process worker — **PASS REBUILT**
  - R7.3 multi-seed stability — **FAIL / ACTIVE**
  - R7.4 larger HU + 3H pilot — TODO after R7.3 diagnostic convergence
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

Historical pre-loss checkpoint: 640 HU roots/seed, cross-seed mean TV 0.3714, p95 TV 0.6878. It remains evidence of the previous implementation, not a direct metric baseline for the rebuilt network until a new same-code multi-seed run is produced.

## Recovery invariants

- No frozen gate may be relaxed.
- `TRUE_HEADS_UP` and `THREE_HANDED` are separate domains for the whole hand.
- A hand starting 3H does not switch strategic domain merely because a player folds/all-ins during that hand.
- Production utility is exact explicit-payout ICM continuation delta; chip delta is diagnostics only.
- Equal-stack simultaneous elimination with unequal unresolved payouts fails closed.
- Every meaningful step is persisted to GitHub `main`.
