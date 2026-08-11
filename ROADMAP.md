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

The confirmed instability chain remains:

```text
Advantage approximation -> nonlinear regret map -> behavior
-> next trajectories -> next strategy targets
```

The first fitted Advantage feedback transition is the first confirmed break: iteration-1 shared strategy targets are identical across seeds, while iteration 2 develops a saturated tail. Support fragmentation and exact shared-state disagreement are both material, so off-support extrapolation is not the sole cause.

Three upstream mechanisms now have independent empirical support:

1. **Advantage policy ensembling** reduces fit/sign variance. Size 8 is the first current-generation 2×128 candidate to clear both frozen cross-seed gates.
2. **Temporal inertia** reduces iteration-to-iteration feedback instability. Size4 temporal w50 materially improved the five-iteration baseline.
3. **Uncertainty-adaptive damping** is now the strongest completed five-iteration mechanism. It damps behavior only where independently fitted regret policies disagree.

## Mandatory durability baseline

Authoritative paired size4 policy mixture:

```text
2×128: mean 0.171940 / p95 0.413605, fits PASS
5×64:  mean 0.266591 / p95 0.567002, fits PASS
```

The five-iteration result is the promotion baseline. Direct 640 escalation remains blocked.

## Completed five-iteration results

All values are 5 CFR iterations × 64 roots = 320 roots/seed.

| candidate | mean TV | p95 TV | fit | status |
|---|---:|---:|---|---|
| size4 no damping | `0.266591` | `0.567002` | PASS | baseline |
| size4 decay tremble e15 | `0.231886` | `0.475154` | PASS | FAIL |
| size4 decay tremble e30 | `0.217853` | `0.457102` | PASS | FAIL |
| size4 decay tremble e45 | `0.211607` | `0.448567` | PASS | FAIL |
| size4 temporal blend w75 | `0.222885` | `0.481673` | PASS | FAIL |
| size4 temporal blend w50 | `0.179915` | `0.395478` | PASS | FAIL |
| size4 first-transition-only e30 | `0.239409` | `0.512631` | PASS | FAIL |
| size4 uncertainty s05 | `0.224640` | `0.471332` | PASS | FAIL |
| **size4 uncertainty s10** | **`0.168098`** | **`0.356780`** | **PASS** | **best completed / FAIL** |
| size1 no damping | `0.438845` | `0.878729` | — | FAIL |
| size1 decay tremble e30 | `0.395333` | `0.798287` | — | FAIL |
| Direct Behavior control | `0.276185` | `0.828670` | PASS | closed |
| Direct Behavior aggregated regret | `0.307350` | `0.914166` | PASS | closed |

The uncertainty-s10 result improves the authoritative five-iteration baseline by about **36.9% in mean TV** and **37.1% in p95 TV**. It misses the frozen gates by only:

```text
mean gap = 0.018098
p95 gap  = 0.006780
```

This supersedes temporal-w50 as the strongest completed durability reference. It is still not an R7.3 pass.

## Size8 short-horizon milestone

Workflow `31440425854`:

```text
2×128
mean TV = 0.139615  PASS <= 0.15
p95 TV  = 0.329689  PASS <= 0.35
fits    = PASS
```

This is the first current-generation short-horizon candidate to clear both frozen cross-seed gates. Because two-iteration wins can decay under feedback depth, size8 still requires 5×64 durability before any acceptance scaling.

Its no-damping durability workflow `31446308103` remains physically in the five-iteration candidate step after build/regression/smoke PASS.

## Promoted compositions under physical test

### Size8 + temporal w50

Workflow `31448623827`:

```text
partial-exact level 2
Advantage policy-mixture size 8
50% current + 50% previous-iteration policy
5×64
```

Build/regression and smoke are PASS; the physical 320-root/seed durability step is running.

### Size8 + uncertainty s10

The newly completed size4 uncertainty-s10 result is sufficiently close to both frozen gates to justify immediate composition with the only short-horizon ensemble size that already cleared them. Commit `cad2a8425a552eb1def4fef5fbca36bc220555ea` added workflow `31449546648`:

```text
partial-exact level 2
Advantage policy-mixture size 8
state-adaptive epsilon = min(0.50, 1.0 * mean-member-TV-to-ensemble-mean)
5×64
```

At the latest physical check this workflow is in build/regression. This is now co-equal highest priority with size8+temporal-w50.

## Remaining active base-matrix candidates

- regret-floor policy mixture size4, epsilon `.05 / .10` — workflow `31444922236`; both physical 5×64 jobs still running;
- size8 no-damping durability — workflow `31446308103`; physical 5×64 running.

The uncertainty-adaptive size4 matrix has completed and no longer belongs in the active set.

## Automatic evidence consolidation

The frozen base matrix remains `SPINCORE_R7_3_DURABILITY_MATRIX_SUMMARY_V4`: 15 candidate rows plus baseline.

Supplemental promoted compositions are consolidated separately by `SPINCORE_R7_3_DURABILITY_EXTENDED_SUMMARY_V2`, which now expects:

```text
17 candidate rows
+ 1 authoritative baseline
= 18 total rows
```

Supplemental rows are `size8_temporal_w50` and `size8_uncertainty_s10`. Ranking is evidence only; no row is automatically promoted to production semantics.

## Residual downstream layer

Final AveragePolicy size4 ensemble remains reserved as a downstream layer:

```text
mean TV = 0.138377
p95 TV  = 0.368730
```

It should be stacked only after a durable upstream winner is selected, not assumed to combine multiplicatively.

## Closed / deprioritized primary branches

- raw root scaling beyond 1280;
- independent x8/x16 path multiplication as standalone fix;
- common-path RNG;
- antithetic x4;
- exhaustive opponent expectation;
- merely raising Advantage optimizer ceiling;
- behavior-aware MSE auxiliary objective;
- exact duplicate aggregation as standalone fix;
- behavior-aware multistart selection;
- raw Advantage ensemble 2/4 standalone;
- legal common-mode centering;
- robust median/trimmed policy aggregation;
- card/suit rewrite as dominant explanation;
- ordinary Direct Behavior as durable solution;
- aggregated-regret Direct Behavior as durable solution;
- direct size4 policy-mixture 640 escalation.

## Promotion rule

Before any mechanism advances to 640 it must:

1. PASS every frozen per-seed fit gate;
2. materially improve both mean and p95 versus `0.266591 / 0.567002` at 5×64;
3. preferably clear `0.15 / 0.35` at the same five-iteration horizon;
4. survive fresh-process reproducibility;
5. have its exact changed behavior semantics frozen/versioned;
6. pass deterministic continuous-vs-stop/restore/continue checkpoint recertification;
7. remain the smallest/interpretable mechanism among statistically comparable winners;
8. keep all frozen gates unchanged.

If a fit-valid size8 composition clears the five-iteration cross-seed gates, the next phase is **semantic freeze + reproducibility + checkpoint/resume recertification**, not immediate 640 scaling.

`READY FOR TABLES = NO`.
