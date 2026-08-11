# R7.3 five-iteration durability program — 2026-08-10

`READY FOR TABLES = NO`. Frozen R7.3 gates remain unchanged.

## Optimization target

The authoritative paired size4 policy mixture is strong over two CFR iterations but degrades at the mandatory five-iteration horizon:

```text
2×128: mean 0.171940 / p95 0.413605
5×64:  mean 0.266591 / p95 0.567002
```

Therefore every upstream candidate must first prove **feedback-depth durability**. No 640 escalation occurs from a short-horizon win alone.

The confirmed causal chain remains:

```text
Advantage approximation -> nonlinear regret map -> behavior
-> next trajectories -> next strategy targets
```

The first fitted Advantage feedback transition is the first observed break, but later replacement of fitted behavior keeps regenerating instability.

## Frozen gates

```text
Advantage weighted NRMSE <= 0.75
AveragePolicy weighted mean TV <= 0.12
cross-seed mean TV <= 0.15
cross-seed p95 TV <= 0.35
```

No gate is relaxed.

## Completed 5×64 results

All rows below are 5 CFR iterations × 64 roots = 320 roots/seed.

| candidate | mean TV | p95 TV | fit | interpretation |
|---|---:|---:|---|---|
| size4 no damping | `0.266591` | `0.567002` | PASS | authoritative durability baseline |
| size4 decay tremble e15 | `0.231886` | `0.475154` | PASS | material improvement |
| size4 decay tremble e30 | `0.217853` | `0.457102` | PASS | stronger |
| size4 decay tremble e45 | `0.211607` | `0.448567` | PASS | strongest global tremble row |
| size4 temporal w75 | `0.222885` | `0.481673` | PASS | material improvement |
| size4 temporal w50 | `0.179915` | `0.395478` | PASS | strong repeated temporal stabilization |
| size4 first-transition e30 | `0.239409` | `0.512631` | PASS | one-shot intervention insufficient |
| size4 uncertainty s05 | `0.224640` | `0.471332` | PASS | adaptive damping helps |
| **size4 uncertainty s10** | **`0.168098`** | **`0.356780`** | **PASS** | **best completed durable row** |
| size1 no damping | `0.438845` | `0.878729` | — | poor |
| size1 decay tremble e30 | `0.395333` | `0.798287` | — | damping helps independently |
| Direct Behavior control | `0.276185` | `0.828670` | PASS | closed as durable solution |
| Direct Behavior aggregated regret | `0.307350` | `0.914166` | PASS | closed; worse than control |

The uncertainty-s10 candidate improves the authoritative five-iteration baseline by approximately:

```text
mean TV improvement = 36.9%
p95 TV improvement  = 37.1%
```

It misses the frozen cross-seed gates by only:

```text
mean gap = 0.018098
p95 gap  = 0.006780
```

This supersedes temporal-w50 as the strongest completed durable mechanism. It is still an R7.3 FAIL because both frozen cross-seed thresholds remain hard requirements.

## What the uncertainty result means

The uncertainty policy uses the disagreement among independently fitted Advantage regret policies at the current state:

```text
disagreement = mean member TV to ensemble-mean policy
epsilon = min(0.50, scale * disagreement)
behavior = (1-epsilon) * ensemble_mean + epsilon * legal_uniform
```

Scale `1.0` is substantially stronger than scale `.50` at five iterations:

```text
s05: mean 0.224640 / p95 0.471332
s10: mean 0.168098 / p95 0.356780
```

The result supports the interpretation that **fitted-policy disagreement is a useful state-local proxy for where feedback should be damped**. Unlike global tremble, stable states are left nearly unchanged.

## Size8 short-horizon milestone

Workflow `31440425854`:

```text
2×128
mean TV = 0.139615  PASS
p95 TV  = 0.329689  PASS
fits    = PASS
```

This is the first current Generation-2 short-horizon candidate to clear both frozen cross-seed gates. Its no-damping 5×64 durability workflow `31446308103` remains physically running after build/regression and smoke PASS.

## Promoted compositions

### Size8 + temporal w50

Workflow `31448623827` combines the strongest short-horizon ensemble size with the strongest earlier repeated-temporal mechanism:

```text
partial-exact level 2
Advantage policy-mixture size 8
feedback = 0.50 current + 0.50 previous iteration
5×64
```

Build/regression and smoke are PASS. The physical 320-root/seed durability step is running.

### Size8 + uncertainty s10

Because uncertainty-s10 is now the strongest completed five-iteration row and size8 is the only short-horizon ensemble size already clearing both gates, commit `cad2a8425a552eb1def4fef5fbca36bc220555ea` added the direct composition:

```text
partial-exact level 2
Advantage policy-mixture size 8
state-adaptive epsilon = min(0.50, 1.0 * mean-member-TV-to-mean)
5×64
```

Workflow `31449546648` passed build/regression and smoke and is now physically executing the five-iteration durability candidate.

This composition is not assumed additive. It must earn its own measured result.

## Remaining active base-matrix experiments

- size8 no-damping durability — `31446308103`;
- regret-floor policy mixture e05/e10 — `31444922236`.

The uncertainty size4 matrix is complete and no longer active.

## Automatic evidence control

The original base consolidator remains frozen as `SPINCORE_R7_3_DURABILITY_MATRIX_SUMMARY_V4`:

```text
15 candidate rows + 1 baseline
```

Supplemental promoted compositions are tracked by `SPINCORE_R7_3_DURABILITY_EXTENDED_SUMMARY_V2`:

```text
17 candidate rows + 1 baseline = 18 rows
supplemental: size8_temporal_w50, size8_uncertainty_s10
```

Ranking is evidence only. It does not modify production semantics or relax acceptance gates.

## Causal conclusions retained

- Advantage approximation/sign variance is material.
- Policy-mixture ensembling is material.
- Global damping has an independent benefit.
- Continued temporal stabilization is stronger than one-shot first-transition damping.
- State-local uncertainty damping is stronger than the completed global and temporal size4 mechanisms.
- Exact shared-state disagreement remains material; off-support extrapolation is not the sole source.
- Ordinary and aggregated-regret Direct Behavior are closed as durable solutions.
- Final AveragePolicy ensembling remains only a downstream residual layer.

## Promotion rule

No candidate moves to 640 merely because it ranks first. Promotion requires:

1. all frozen per-seed fit gates PASS;
2. material improvement in both mean and p95 versus `0.266591 / 0.567002` at 5×64;
3. preferably full `0.15 / 0.35` cross-seed clearance at five iterations;
4. fresh-process reproducibility;
5. explicit freeze/versioning of changed behavior semantics;
6. deterministic continuous-vs-stop/restore/continue checkpoint recertification;
7. smallest/interpretable winner among statistically comparable mechanisms;
8. frozen gates unchanged.

If a fit-valid size8 composition clears the five-iteration gates, the immediate next stage is **semantic freeze + fresh-process reproducibility + checkpoint/resume recertification**. Acceptance-scale 640 follows only after those checks.
