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

The first fitted Advantage feedback transition is the first confirmed break. Support fragmentation and exact shared-state disagreement remain material, so off-support extrapolation is not the sole cause.

Three upstream mechanisms have independent empirical support:

1. **Advantage policy ensembling** reduces fit/sign variance. Size8 is the first current-generation 2×128 candidate to clear both frozen cross-seed gates.
2. **Temporal inertia** reduces iteration-to-iteration feedback instability.
3. **Uncertainty-adaptive damping** is the strongest completed five-iteration mechanism and damps only states where independently fitted regret policies disagree.

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

Uncertainty-s10 improves the baseline by about **36.9% mean** and **37.1% p95**, but still misses the frozen gates by:

```text
mean gap = 0.018098
p95 gap  = 0.006780
```

It is therefore the current durable reference, not an R7.3 pass.

## Size8 short-horizon milestone

Workflow `31440425854`:

```text
2×128
mean TV = 0.139615  PASS <= 0.15
p95 TV  = 0.329689  PASS <= 0.35
fits    = PASS
```

This is the first current-generation short-horizon candidate to clear both frozen cross-seed gates. It still requires full 5×64 durability.

## Current physical candidate program

Every listed job has already passed build/regression and smoke unless otherwise noted.

### Size8 no damping

Workflow `31446308103`, job `93641202513`:

```text
Run physical size8 five-iteration durability — IN PROGRESS
```

### Regret-floor size4 e05/e10

Workflow `31444922236`, jobs `93636932879` / `93636932877`:

```text
Run physical five-iteration regret-floor candidate — IN PROGRESS
```

### Size8 + temporal w50

Workflow `31448623827`, job `93648112606`:

```text
Run physical size8 temporal-w50 five-iteration durability — IN PROGRESS
```

### Size8 + uncertainty s10

Workflow `31449546648`, job `93650858760`:

```text
partial-exact level 2
Advantage policy-mixture size8
state-adaptive epsilon = min(0.50, 1.0 * mean-member-TV-to-mean)
5×64
Run physical size8 uncertainty-s10 five-iteration durability — IN PROGRESS
```

### Local uncertainty calibration — size4 s1.25 / s1.50

Because size4 s1.0 is already extremely close to the frozen p95 gate, workflow `31450032347` tests the smallest plausible further intervention before preferring a more expensive size8 composition:

```text
size4, cap 0.50
scale 1.25
scale 1.50
5×64
```

Jobs `93652314342` and `93652314379` passed build/regression and smoke and are both physically executing `Run physical five-iteration uncertainty extension`.

If a smaller size4 scale clears both gates, it is preferred over a statistically comparable size8 composition because it is simpler and cheaper.

## Candidate checkpoint/resume readiness

R7.2 already certifies `SPINCORE_R7_CHECKPOINT_V2` for the single-primary-network path, but R7.3 ensembles contain extra state that the plain `DomainBundle` does not carry.

`python/spincore/r7_candidate_checkpoint.py` now defines:

```text
SPINCORE_R7_CANDIDATE_BEHAVIOR_V1
```

as an extra payload layered on the unchanged base checkpoint. It serializes side Advantage members, previous temporal members, wrapper parameters and fit generation while reusing the authoritative restored primary model as ensemble member zero and failing closed on a primary-state mismatch.

Main regression `31449980549` passed after this preparation:

```text
C++ regression PASS
Python 32 passed
```

This is serialization readiness only. The winning exact behavior still requires a physical continuous-vs-stop/restore/continue recertification after winner selection and before 640.

## Automatic evidence consolidation

The frozen base matrix remains `SPINCORE_R7_3_DURABILITY_MATRIX_SUMMARY_V4`:

```text
15 candidate rows + 1 baseline
```

The supplemental consolidator is now `SPINCORE_R7_3_DURABILITY_EXTENDED_SUMMARY_V3`:

```text
19 candidate rows + 1 baseline = 20 total rows
```

Supplemental rows:

- `size4_uncertainty_s125`
- `size4_uncertainty_s150`
- `size8_temporal_w50`
- `size8_uncertainty_s10`

Ranking is evidence only; it never relaxes a gate or promotes production semantics automatically.

## Residual downstream layer

Final AveragePolicy size4 ensemble remains reserved as a downstream layer:

```text
mean TV = 0.138377
p95 TV  = 0.368730
```

It is tested only after an upstream durable winner is identified; its gain is not assumed additive.

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
3. clear or explicitly solve any remaining miss to `0.15 / 0.35` at the same five-iteration horizon;
4. survive fresh-process reproducibility;
5. have exact changed behavior semantics frozen/versioned;
6. pass deterministic continuous-vs-stop/restore/continue candidate checkpoint recertification;
7. remain the smallest/interpretable mechanism among statistically comparable winners;
8. keep all frozen gates unchanged.

A five-iteration gate-clearing winner moves next to **semantic freeze + fresh-process reproducibility + checkpoint/resume recertification**, not directly to 640.

`READY FOR TABLES = NO`.
