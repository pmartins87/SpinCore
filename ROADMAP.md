# SpinCore finite roadmap — recovery integrity corrected 2026-08-09

The final endpoint remains **ready to start using at the tables**. `READY FOR TABLES = NO` until every required gate passes.

## Historical/recovery evidence

- R0 Foundation / canonical repository — PASS evidence
- R1 Complete poker engine — PASS evidence
- R2 Canonical infoset + neural encoder — PASS evidence
- R3 Tournament continuation value (`ICM_EXACT_V1`, explicit payout) — PASS evidence
- R4 Neural infrastructure — PASS evidence
- R5 CFR correctness oracle — PASS evidence
- R6 Deep CFR integration — PASS evidence
- R7.0 instrumentation — PASS evidence
- R7.1 native own-reach frontier — PASS evidence
- R7.2 full-reservoir audit / LCFR weighting / resume / worker — PASS evidence
- R7.3 multi-seed stability — FAIL at last historical checkpoint

## Current physical gate

Before R7.3 can continue, **RECOVERY_INTEGRITY_REPAIR** must PASS. The current `main` is not self-contained: its build graph references source files that are absent, while the prior recertification report lists R7 files that were never persisted into the reachable tree. Therefore the old labels `PASS_RECERTIFIED` are retained only as historical evidence, not as a claim that the present checkout can reproduce 76 C++ + 47 Python tests.

Required repair sequence:

1. Restore the missing R0–R7.2 source/test tree from a recoverable bundle, Git object, or other authoritative source.
2. Verify the restored files against preserved hashes wherever available.
3. Build Release and sanitizer configurations from a clean checkout.
4. Run the full C++ and Python regression suites.
5. Re-certify R6/R7.0–R7.2 only if the physical checkout reproduces the gates.
6. Resume R7.3 at materially larger HU self-play with the frozen gates unchanged.

## Frozen R7.3 gates

- Advantage weighted normalized RMSE <= 0.75
- Average-policy weighted mean TV <= 0.12
- Cross-seed mean TV <= 0.15
- Cross-seed p95 TV <= 0.35

Historical last HU checkpoint: 640 roots/seed, cross-seed mean TV 0.3714, p95 TV 0.6878. R7.3 remains FAIL.

After R7:

- R8 Production training — TODO
- R9 Strategic audit — TODO
- R10 OpenHoldem runtime — TODO
- R11 Safe exploitation — TODO
- R12 Operational homologation — TODO

No frozen gate may be relaxed. `TRUE_HEADS_UP` and `THREE_HANDED` remain separate strategic domains. Production utility remains exact explicit-payout ICM continuation delta. Material recovery work is persisted to `main`; no recovery branch.
