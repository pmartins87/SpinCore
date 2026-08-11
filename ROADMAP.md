# SpinCore finite roadmap — canonical recovery generation 2

Final endpoint: **ready to start using at the tables**. `READY FOR TABLES = NO` until every required gate passes.

- R0 Foundation / canonical repository — **PASS REBUILT**
- R1 Complete poker engine — **PASS REBUILT**
- R2 Canonical infoset + neural encoder — **PASS REBUILT**
- R3 Tournament continuation value (`ICM_EXACT_V1`, explicit payout) — **PASS REBUILT**
- R4 Neural infrastructure — **PASS REBUILT**
- R5 CFR correctness oracle — **PASS REBUILT**
- R6 Deep CFR integration on authoritative `SpinTraversalState` — **PASS REBUILT**
- R7 Pilot / performance / statistical stability — **IN PROGRESS**
  - R7.0 approximation metrics / full-reservoir audit — **PASS REBUILT**
  - R7.1 native own-reach frontier — **PASS REBUILT**
  - R7.2 LCFR weighting / exact checkpoint+resume / fresh-process worker — **PASS REBUILT**
  - R7.3 multi-seed stability — **FAIL / ACTIVE**
  - R7.4 larger HU + 3H pilot — TODO after R7.3 convergence
- R8 Production training — TODO
- R9 Strategic audit — TODO
- R10 OpenHoldem runtime — TODO
- R11 Safe exploitation — TODO
- R12 Operational homologation — TODO

## Frozen R7.3 gates

- Advantage weighted normalized RMSE `<= 0.75`
- AveragePolicy weighted mean TV `<= 0.12`
- cross-seed mean TV `<= 0.15`
- cross-seed p95 TV `<= 0.35`

No gate is relaxed.

## Authoritative acceptance contract

```text
deck_seed = seed * 1_000_003 + global_root * 97 + iteration
```

`global_root` is continuous across CFR iterations. The recovered path preserves one persistent live `bundle.batch_rng` through primary collection/training in execution order unless a diagnostic explicitly declares otherwise.

## R7.3 causal state

The first fitted Advantage feedback transition is the confirmed break:

```text
iteration 1 shared strategy targets: mean/p95 TV = 0 / 0
iteration 2 shared strategy targets: mean/p95 TV = 0.473946 / 1.0
```

Known chain:

```text
Advantage approximation -> nonlinear regret map -> behavior
-> next trajectories -> next strategy targets
```

Support fragmentation and Advantage fit variance are material. Exact shared observations also retain substantial policy disagreement, so off-support extrapolation is not the sole tail source.

## Current durability baseline

The paired size-4 policy mixture is reproducibly strong at two iterations:

```text
2×128: mean 0.171940 / p95 0.413605, fits PASS
```

A fresh physical rerun reproduced those cross-seed values exactly.

But at five iterations:

```text
5×64: mean 0.266591 / p95 0.567002, fits PASS
```

Therefore the size-4 640 workflow remains **DORMANT**. No candidate advances to 640 until it materially improves both five-iteration metrics.

## Completed branches in the current durability phase

### Ordinary Direct Behavior — CLOSED as durable solution

Short horizon:

```text
2×128: mean 0.142553 / p95 0.426860
```

Five-iteration physical result, workflow `31441650915`, evidence `db8fe87f6fd07ca651825436173e2a1f01b89d40`:

```text
5×64: mean 0.276185
      p50 0.205417
      p95 0.828670
      max 0.997361
      frozen per-seed fits PASS
```

The smooth surrogate's apparent short-horizon benefit is not durable; its p95 becomes far worse than the size-4 durability baseline. Ordinary Direct Behavior remains useful only as causal evidence and is closed as a production/durability path.

### Final AveragePolicy ensemble — useful residual layer

After size-4 policy-mixture CFR at 2×128:

| final policy members | mean TV | p95 TV |
|---:|---:|---:|
| 1 | `0.179750` | `0.434644` |
| 2 | `0.159165` | `0.404792` |
| 4 | **`0.138377`** | **`0.368730`** |

All fits PASS. The size-4 final ensemble crosses the mean gate and misses p95 by only `0.01873`, proving a meaningful downstream residual layer. It will be stacked only after an upstream mechanism first proves five-iteration durability; we do not spend a separate 5×64 run on it in isolation.

### Support-conditioned tail — off-support-only explanation rejected

At paired 2×128:

```text
union A+B:                 mean 0.167806 / p95 0.411491
exact shared observations: mean 0.181562 / p95 0.398769
A-only exact support:       mean 0.187291 / p95 0.417470
B-only exact support:       mean 0.157756 / p95 0.418203
```

Shared-state disagreement is itself material.

### Robust member aggregation — CLOSED

Median/trimmed policy aggregation worsened p95 relative to ordinary probability averaging. Rare single-member outliers are not the main mechanism.

### Size-1 temporal damping factorial

At 5×64:

```text
size1 no damping:       mean 0.438845 / p95 0.878729
size1 epsilon0=.30:     mean 0.395333 / p95 0.798287
```

Damping independently improves both metrics by roughly 9–10%, while ensembling remains the larger effect. The active size-4 epsilon matrix determines their interaction.

## Regret-target order and sign-boundary evidence

### Aggregated-regret target order

Same-memory diagnostic, workflow `31441852607`:

```text
10,565 raw Advantage samples
10,515 unique exact observation/legal groups
```

Independent surrogate disagreement:

| target order | mean TV | p95 TV |
|---|---:|---:|
| mean of sample-level RM | `0.104576` | `0.357100` |
| **RM(LCFR-weighted mean regret)** | **`0.102435`** | **`0.347652`** |

The regret-first order marginally clears the same-memory p95 gate. Because duplicate exact observations are rare, this is interpreted as a target-order clue rather than a compression result. Workflow `31444324235` is physically testing this construction at both 2×128 and 5×64. It remains non-equivalent/non-promotable without semantic review.

### Near-zero regret sign fragility

The prior same-memory sign-sensitivity experiment showed material hard-regret-map instability: independent fits disagree in legal regret sign/support, and a scale-relative positive-regret floor reduces pairwise policy TV post hoc.

That evidence now has a direct 5×64 E2E test: workflow `31444922236` runs size-4 policy mixtures with:

```text
floor = epsilon * RMS(legal predicted advantages)
epsilon = 0.05 / 0.10
weight_a = max(advantage_a, 0) + floor
```

Each member is regularized before probability averaging. This acts exactly at the demonstrated nonlinear sign boundary rather than globally flattening all states. It is still an explicit algorithm change and cannot become production semantics without versioning/recertification.

## Active five-iteration matrix

Current physical experiments:

- global decaying tremble size4: `epsilon0 0.15 / 0.30 / 0.45`, decay `.50` — `31441018067`;
- previous-policy temporal blend size4: current weight `.50 / .75` — `31441224117`;
- first-transition-only tremble size4: `[.30,0,0,0]` — `31441567261`;
- uncertainty-adaptive damping size4: disagreement scale `.50 / 1.00`, cap `.50` — `31442367579`;
- aggregated-regret Direct Behavior: 2×128 + 5×64 — `31444324235`;
- regret-floor policy mixture size4: epsilon `.05 / .10` — `31444922236`;
- fresh 320 size4 reproducibility copy — `31440366909`;
- size8 policy mixture 2×128 — `31440425854`; any short-horizon win still needs its own 5×64 durability gate.

## Automatic evidence control

Durability consolidation is `SPINCORE_R7_3_DURABILITY_MATRIX_SUMMARY_V3`. It waits for 14 candidate rows plus the authoritative baseline and ranks only fit-valid candidates that improve both baseline metrics.

The workflows use `workflow_run` completion triggers rather than relying on evidence-file pushes made by `GITHUB_TOKEN`, because GitHub suppresses recursive workflow triggering from such pushes. The same fix was applied to fresh-run reproducibility consolidation.

Direct Behavior variants are reported but excluded from conservative automatic promotion because theoretical equivalence is not established.

## Promotion rule

Before any mechanism advances to 640 it must:

1. PASS every frozen per-seed fit gate;
2. materially improve **both** mean and p95 versus `0.266591 / 0.567002` at 5×64;
3. survive fresh-run reproducibility checks;
4. be the smallest/interpretable mechanism among statistically comparable candidates;
5. have changed behavior semantics explicitly frozen/versioned;
6. pass deterministic continuous-vs-stop/restore/continue checkpoint recertification;
7. keep frozen gates unchanged.

If an upstream durable winner approaches but does not clear the frozen cross-seed gates, the already-proven final AveragePolicy size-4 residual ensemble is the next layer to stack.

## Closed / deprioritized primary branches

- raw root scaling beyond 1280;
- independent x8/x16 path multiplication as a standalone fix;
- common-path RNG;
- antithetic x4;
- exhaustive opponent expectation;
- merely raising Advantage optimizer ceiling;
- behavior-aware MSE auxiliary objective;
- exact duplicate aggregation as standalone fix;
- behavior-aware multistart selection;
- raw Advantage ensemble 2/4 standalone;
- final AveragePolicy ensemble standalone;
- legal common-mode centering;
- robust median/trimmed aggregation;
- card/suit rewrite as dominant explanation;
- ordinary Direct Behavior as a durable solution;
- direct size4 policy-mixture 640 escalation.

`READY FOR TABLES = NO`.
