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
    - corrected 640-root fit gates — **PASS**, cross-seed — **FAIL**
    - 1280-root scale — **FAIL**, cross-seed essentially flat and policy fit degraded
    - controlled 640-root variance decomposition — **DONE: CFR-memory variance dominant**
    - shared-deck + support-conditioned diagnostic — **RUNNING**
    - brute-force root scaling — **PAUSED until causal diagnosis closes**
  - R7.4 larger HU + 3H pilot — TODO after R7.3 convergence
- R8 Production training — TODO
- R9 Strategic audit — TODO
- R10 OpenHoldem runtime — TODO
- R11 Safe exploitation — TODO
- R12 Operational homologation — TODO

## Current R7.3 evidence

Corrected 640 roots/seed:

- seed 20260829 Advantage NRMSE `0.65364`, policy TV `0.10571` — fit gates PASS
- seed 20260807 Advantage NRMSE `0.67165`, policy TV `0.11697` — fit gates PASS
- cross-seed mean TV `0.47765` — FAIL
- cross-seed p95 TV `0.90240` — FAIL

1280 roots/seed:

- cross-seed mean TV `0.47319`
- cross-seed p95 TV `0.87528`
- both Advantage fits still PASS
- both AveragePolicy fits exceeded `0.12`

The 640-root controlled variance decomposition then fit two controlled AveragePolicy replicas to each CFR memory. All four policy-fit gates passed, but changing CFR memory while holding policy optimizer/init seed fixed produced mean TV `0.46989`, versus `0.24265` average disagreement between replicas trained on the same memory. Ratio: `1.93647`. Persisted diagnosis: `CFR_MEMORY_VARIANCE_DOMINANT`.

The same-memory figure is not treated as pure optimizer variance because that comparison used a union of both memories and therefore includes off-support extrapolation. The active follow-up uses a shared deck stream and evaluates each comparison separately on memory-A support, memory-B support, and their union.

## Frozen R7.3 gates

- Advantage weighted normalized RMSE <= 0.75
- Average-policy weighted mean TV <= 0.12
- Cross-seed mean TV <= 0.15
- Cross-seed p95 TV <= 0.35

Historical pre-loss checkpoint: 640 HU roots/seed, cross-seed mean TV 0.3714, p95 TV 0.6878. It remains evidence of the previous implementation, not a direct metric baseline for the rebuilt network.

## Recovery invariants

- No frozen gate may be relaxed.
- `TRUE_HEADS_UP` and `THREE_HANDED` are separate domains for the whole hand.
- A hand starting 3H does not switch strategic domain merely because a player folds/all-ins during that hand.
- Production utility is exact explicit-payout ICM continuation delta; chip delta is diagnostics only.
- Equal-stack simultaneous elimination with unequal unresolved payouts fails closed.
- Every meaningful step is persisted to GitHub `main`.
- Root scaling is not resumed while a lower-cost causal diagnostic can distinguish approximation, chance-coverage, and CFR-dynamics failures.
