# SpinCore recovery generation 2 — R0 through R7.2 physical recertification

Date: 2026-08-09

## Why this rebuild exists

The prior recovery accidentally persisted status/evidence while omitting much of the physical R0-R5 source and several R7 files. The original R5 Git object `445fb4a56da69f9a3ca48acbe13a6cfb95055ac7` is not present in GitHub and the original ZIP/bundle is no longer retrievable from the current File Library. The project therefore rebuilt the preserved semantic contracts instead of pretending the sparse checkout was reproducible.

This generation is **not byte-for-byte identical** to the lost R5 checkout. It is a new physically self-contained implementation constrained by the preserved roadmap, ABI, utility semantics, action abstraction, and R6/R7 gates.

## Physical validation

- C++ Release: 47 PASS / 0 FAIL
- C++ ASan + UBSan: 47 PASS / 0 FAIL
- Python full suite: 23 PASS / 0 FAIL
- Solver ABI: `SPINCORE_SOLVER_C_ABI_V2`

## Contracts recertified

- deterministic 52-card deal and exact hidden-state cloning;
- correct best-five-of-seven evaluator including wheel straights and tie breakers;
- NLHE 2/3-player betting, blinds, full raises, all-ins, street progression, side pots and chip conservation;
- true heads-up blind/action order and 3H topology kept as separate strategic domains;
- canonical actor infoset never includes opponent private cards;
- `SPNNIV1` fixed binary neural payload;
- exact ICM continuation values with explicit `[1st,2nd,3rd]` payout vector;
- post-elimination HU keeps the already locked third-place payout;
- equal-stack simultaneous elimination under unequal unresolved payouts fails closed;
- external-sampling Deep CFR advantage targets are centered as `V(I,a)-sum sigma(a)V(I,a)`;
- Average Policy uses own-reach collection;
- Algorithm-R reservoirs preserve LCFR iteration weights;
- native C++ frontier enumerates non-target branches until target actor or terminal with node/depth caps;
- native own-reach collection matches Python semantic collection in deterministic regression;
- audit samples are deterministic and stratified across the full reservoir rather than prefix-capped;
- weighted Advantage nRMSE and Average Policy TV metrics are implemented separately;
- checkpoint persists model, optimizer, reservoirs, Python RNG, torch RNG, counters and mid-iteration progress;
- continuous execution equals save/destroy/restore/continue exactly in deterministic regression;
- a fresh-process optimizer worker reloads and atomically rewrites the checkpoint.

## R7.3

R7.3 remains FAIL/active. Frozen gates remain unchanged. The old 640-roots/seed `mean TV=0.3714`, `p95=0.6878` result belongs to the pre-loss implementation and is retained only as historical evidence; a fresh rebuilt-code two-seed baseline is now required before scaling.

`READY FOR TABLES = NO`.
