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

Two upstream mechanisms are now independently demonstrated:

1. **Advantage policy ensembling** reduces same-memory approximation/sign variance. Size 8 is the first current-generation short-horizon candidate to clear both frozen cross-seed gates.
2. **Temporal inertia** reduces feedback-depth instability. A 50/50 current-vs-previous policy blend is the strongest completed five-iteration mechanism so far.

Support fragmentation and exact shared-state disagreement remain material; off-support extrapolation is not the sole tail source.

## Mandatory durability baseline

Authoritative paired size-4 policy mixture:

```text
2×128: mean 0.171940 / p95 0.413605, fits PASS
5×64:  mean 0.266591 / p95 0.567002, fits PASS
```

The five-iteration result is the relevant promotion baseline. Direct 640 escalation remains blocked.

## Completed five-iteration matrix results

All values below are 5 CFR iterations × 64 roots = 320 roots/seed.

| candidate | mean TV | p95 TV | status |
|---|---:|---:|---|
| size4 no damping | `0.266591` | `0.567002` | baseline |
| size4 decay tremble e15 | `0.231886` | `0.475154` | FAIL |
| size4 decay tremble e30 | `0.217853` | `0.457102` | FAIL |
| size4 decay tremble e45 | `0.211607` | `0.448567` | FAIL |
| size4 temporal blend w75 | `0.222885` | `0.481673` | FAIL |
| **size4 temporal blend w50** | **`0.179915`** | **`0.395478`** | **best completed / FAIL** |
| size4 first-transition-only e30 | `0.239409` | `0.512631` | FAIL |
| size1 no damping | `0.438845` | `0.878729` | FAIL |
| size1 decay tremble e30 | `0.395333` | `0.798287` | FAIL |
| Direct Behavior control | `0.276185` | `0.828670` | closed |
| Direct Behavior aggregated regret | `0.307350` | `0.914166` | closed |

The temporal w50 candidate improves the authoritative size4 durability baseline by about **32.5% in mean TV** and **30.3% in p95 TV**, but still misses the frozen gates by approximately:

```text
mean gap = 0.029915
p95 gap  = 0.045478
```

This makes temporal inertia the strongest durable causal result, but not yet an R7.3 pass.

## Size8 short-horizon milestone

Workflow `31440425854`, evidence commit `bfe6d4845600c3eafed36c85c0113756763f6910`:

```text
2×128
mean TV = 0.139615  PASS <= 0.15
p95 TV  = 0.329689  PASS <= 0.35
fits    = PASS
```

This is the first current-generation short-horizon candidate to clear both frozen cross-seed gates. It is **not** promoted to 640 because size4 already proved that two-iteration success can decay badly under five feedback iterations.

Its mandatory 5×64 durability workflow is `31446308103` and is physically running after build/regression and smoke PASS.

## Strongest composition now under test

The two best demonstrated mechanisms target different parts of the failure:

- size8: approximation/sign variance;
- temporal w50: iteration-to-iteration feedback instability.

Therefore commit `8166871908f0580cd3170ebd038ca0ad83072951` added a direct composition:

```text
partial-exact level 2
Advantage policy-mixture size 8
50% current policy + 50% previous-iteration policy
5 iterations × 64 roots
320 roots/seed
```

Workflow `31448623827` is active. At the latest physical check, build/regression had passed and the size8 temporal-w50 smoke was running. This composition is supplemental to the original 15-candidate durability matrix and is the highest-priority follow-up because it combines the strongest short-horizon and strongest durable mechanisms without relaxing any gate.

## Remaining active base-matrix candidates

- regret-floor policy mixture size4, epsilon `.05 / .10` — workflow `31444922236`; both physical 5×64 jobs running after build/smoke PASS;
- uncertainty-adaptive damping size4, scale `.50 / 1.00`, cap `.50` — workflow `31442367579`; both physical 5×64 jobs running after build/smoke PASS;
- size8 no-damping durability — workflow `31446308103`; physical 5×64 running.

The base durability consolidator is `SPINCORE_R7_3_DURABILITY_MATRIX_SUMMARY_V4`: 15 candidate rows plus the authoritative baseline. At the latest audit, 10 candidate rows were complete and 5 were still pending. The size8+temporal-w50 composition is intentionally outside that frozen base matrix and will be compared separately against its winner.

## Completed/closed branches

### Ordinary Direct Behavior — CLOSED

It improved the short horizon but deteriorated badly at five iterations. The aggregated-regret variant was worse again. Both remain causal controls only; their algorithmic semantics are not production-equivalent Deep CFR behavior.

### Final AveragePolicy ensemble — residual layer only

After size4 policy-mixture CFR at 2×128:

| final policy members | mean TV | p95 TV |
|---:|---:|---:|
| 1 | `0.179750` | `0.434644` |
| 2 | `0.159165` | `0.404792` |
| 4 | **`0.138377`** | **`0.368730`** |

The size4 final ensemble is useful downstream, but it will only be stacked after an upstream mechanism proves five-iteration durability.

### Other closed/deprioritized primary branches

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
- direct size4 policy-mixture 640 escalation.

## Promotion rule

Before any mechanism advances to 640 it must:

1. PASS every frozen per-seed fit gate;
2. materially improve **both** mean and p95 versus `0.266591 / 0.567002` at 5×64;
3. preferably clear the frozen cross-seed gates at the same five-iteration horizon; if only narrowly outside, any downstream residual layer must be tested explicitly rather than assumed;
4. survive fresh-process reproducibility checks;
5. be the smallest/interpretable mechanism among statistically comparable candidates;
6. have changed behavior semantics explicitly frozen/versioned;
7. pass deterministic continuous-vs-stop/restore/continue checkpoint recertification;
8. keep frozen gates unchanged.

If size8+temporal-w50 clears or nearly clears both gates, the next phase is **semantic freeze + fresh-process reproducibility + checkpoint/resume recertification**, not immediate 640 scaling.

`READY FOR TABLES = NO`.
