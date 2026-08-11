# R7.4 finite held-out HU + 3H gate — precommitted 2026-08-11

`READY FOR TABLES = NO`.

This gate is defined **before any R7.4 strategic/stability result is observed**. Its purpose is to prevent post-result threshold, seed, domain, or scale selection. R7.4 remains blocked until the corrected R7.3 certification chain passes fresh reproducibility, checkpoint/resume recertification, and exact-source 640 acceptance.

## Accepted R7.3 mechanism carried into R7.4

R7.4 does not search for a new algorithm. It evaluates the deliberately selected R7.3 winner:

```text
behavior semantic: SPINCORE_R7_3_UNCERTAINTY_POLICY_MIXTURE_V1
ensemble size: 4
epsilon scale: 1.75
epsilon cap: 0.50
partial-exact opponent levels: 2
primary RNG: one persistent live bundle.batch_rng
utility: exact explicit-payout ICM delta
deck formula: seed*1000003 + global_root*97 + iteration
```

The exact accepted R7.3 source commit is used. R7.4 harness code is overlaid into a detached accepted-source worktree and must not replace algorithm modules.

## Held-out seeds

R7.4 must not reuse the R7.3 selection seeds `20260829, 20260807`.

Two held-out seeds are derived mechanically from the immutable frozen winner evidence SHA-256:

```text
SHA256("SpinCore|R7.4|heldout|index|" + frozen_evidence_sha256)
```

The first positive 31 bits are used, rejecting zero, duplicates, and collisions with the R7.3 seeds. The same held-out pair is used for HU, the 3H screen, and the 3H confirmation. This seed rule is frozen before results.

## Domain scenarios

### TRUE_HEADS_UP

Six deterministic scenario variants:

```text
stacks (0,750,750), (0,500,1000), (0,1000,500)
× dealer seat 1 or 2
```

Seat 0 remains dead and the whole-hand domain remains true HU.

### THREE_HANDED

Fifteen deterministic variants:

```text
stack profiles:
(500,500,500)
(250,500,750)
(250,750,500)
(500,250,750)
(750,250,500)
× dealer seat 0, 1, or 2
```

All profiles preserve the same 1500-chip tournament pool. Scenarios are assigned by `global_root % scenario_cycle_size`; every scenario must be exercised.

## Frozen statistical gates

R7.4 reuses the existing R7.3 model-fit and cross-seed stability thresholds rather than inventing easier domain-specific thresholds:

```text
Advantage weighted NRMSE <= 0.75
AveragePolicy weighted mean TV <= 0.12
cross-seed mean TV <= 0.15
cross-seed p95 TV <= 0.35
```

All per-seed fit gates, scenario coverage, and cross-seed gates must pass.

## Finite execution sequence

The R7.4 gate is finite and precommitted:

1. **Structural accepted-source HU/3H preflight** — topology/domain identities, clone/neural exactness, chip conservation, exact ICM zero-sum checks. Structural PASS does not imply strategic PASS.
2. **Held-out HU confirmation** — 5 CFR iterations × 128 roots = **640 roots/held-out seed**.
3. **Held-out 3H cost-bounded screen** — 5 × 64 = **320 roots/held-out seed**.
4. Only if both 2 and 3 pass, **held-out 3H confirmation** — 5 × 128 = **640 roots/held-out seed** with the exact same held-out seeds and scenario rule.
5. R7.4 PASS requires HU640 PASS + 3H320 PASS + 3H640 PASS. No partial PASS is sufficient.

The 3H 320 screen is only a compute filter. It cannot itself complete R7.4; the full 3H 640 confirmation is mandatory after a screen pass.

## Evidence schemas

```text
SPINCORE_R7_4_DOMAIN_PREFLIGHT_V1
SPINCORE_R7_4_HELDOUT_DOMAIN_STABILITY_V1
SPINCORE_R7_4_HELDOUT_SCREEN_SUMMARY_V1
SPINCORE_R7_4_FINAL_GATE_V1
```

Canonical evidence files:

```text
validation/R7_4_DOMAIN_PREFLIGHT.json
validation/R7_4_HELDOUT_HU_640.json
validation/R7_4_HELDOUT_3H_320.json
validation/R7_4_HELDOUT_SCREEN_SUMMARY.json
validation/R7_4_HELDOUT_3H_640_CONFIRMATION.json
validation/R7_4_FINAL_GATE.json
```

## Fail-closed rules

Any of the following blocks R7.4:

- corrected R7.3 640 certification is absent or provenance-incomplete;
- exact accepted source cannot be proven;
- R7.3 selection seeds are reused;
- scenario coverage is incomplete;
- side ensemble members perturb primary RNG;
- deck or partial-exact contract changes;
- any frozen fit or cross-seed gate fails;
- held-out HU or either 3H stage fails;
- a result is produced without the accepted-source worktree/worker-overlay provenance.

R7.4 PASS authorizes only R8 production training. It **does not** set table readiness. R8–R12 remain mandatory.
