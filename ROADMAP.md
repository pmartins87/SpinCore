# SpinCore finite roadmap — recovery continuation

The final endpoint is **ready to start using at the tables**. `READY FOR TABLES = NO` until every required gate passes.

- R0 Foundation / canonical repository — PASS
- R1 Complete poker engine — PASS
- R2 Canonical infoset + neural encoder — PASS
- R3 Tournament continuation value — PASS
- R4 Neural infrastructure — PASS
- R5 CFR correctness oracle — PASS
- R6 Deep CFR integration on authoritative `SpinTraversalState` — RECOVERING / physical traversal state re-materialized and regression-tested
- R7 Pilot / performance / statistical stability — historical R7.0–R7.2 PASS; R7.3 stability gate still FAIL; physical stack being re-materialized
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

Historical last HU checkpoint: 640 roots/seed, mean TV 0.3714, p95 TV 0.6878. Therefore R7.3 remains FAIL until physically revalidated at scale.

## Recovery rule

No historical PASS is silently downgraded, but no historical result is treated as physically reproducible until its code/test evidence is re-materialized in this repository. No gate may be relaxed to accelerate recovery.
