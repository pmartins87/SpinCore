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
    - shared-deck + support-conditioned diagnostic — **DONE: common deck stream barely changes divergence; off-support extrapolation material**
    - exact/card-isomorphic support-overlap diagnostic — **RUNNING**
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

The 640-root controlled variance decomposition fit two controlled AveragePolicy replicas to each CFR memory. All four policy-fit gates passed, but changing CFR memory while holding policy optimizer/init seed fixed produced mean TV `0.46989`, versus `0.24265` average disagreement between replicas trained on the same memory. Ratio: `1.93647`. Persisted diagnosis: `CFR_MEMORY_VARIANCE_DOMINANT`.

The support-conditioned shared-deck follow-up then held the root deal/future-board stream identical across the two algorithm seeds. Across-memory mean TV on the union was still `0.46007`, versus `0.46989` in the independent-deck reference; ratio `0.97909`. Thus card/chance-stream variation at the root explains only a small fraction of the observed instability. The classifier remained conservative because one controlled policy fit finished slightly above the frozen `0.12` gate.

Support conditioning was more informative: same-memory replica disagreement averaged `0.12031` on the memory's own support but `0.29231` on the other memory's support, a `2.42959x` ratio. Persisted support diagnosis: `OFF_SUPPORT_POLICY_EXTRAPOLATION_MATERIAL`.

The active follow-up therefore measures the **memory support itself**, without relying on neural predictions: exact observation overlap, LCFR-weighted target disagreement on the exact intersection, overlap after global suit relabeling, and overlap after also removing private-card order and flop-card order. This is diagnostic only; it does not change `SPNNIV1` or any acceptance gate. If poker card isomorphism substantially expands the shared support, a versioned canonical-card representation becomes the leading correction. If it does not, traversal/action RNG coupling and CFR path-support variance are next.

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
- Root scaling is not resumed while a lower-cost causal diagnostic can distinguish approximation, chance-coverage, representation, and CFR-dynamics failures.
