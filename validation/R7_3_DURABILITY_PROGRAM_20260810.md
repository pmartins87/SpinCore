# R7.3 five-iteration durability program — 2026-08-10

`READY FOR TABLES = NO`. Frozen R7.3 gates remain unchanged.

## Optimization target

The authoritative paired size-4 policy mixture was strong over two CFR iterations (`mean TV 0.171940`, `p95 0.413605`) but deteriorated over the mandatory 5×64 compounding horizon:

```text
mean TV = 0.2665907145
p50 TV  = 0.2468046695
p95 TV  = 0.5670017600
max TV  = 0.9055466652
fit gates = PASS
```

Short-horizon variance reduction is therefore insufficient for promotion. Every upstream candidate must first prove **feedback-depth durability**.

The causal forensic independently localizes the first break: exact shared strategy targets are identical in iteration 1 and become approximately `mean TV 0.473946 / p95 1.0` after the first fitted Advantage behavior feeds back into collection.

## Frozen gates

```text
Advantage weighted NRMSE <= 0.75
AveragePolicy weighted mean TV <= 0.12
cross-seed mean TV <= 0.15
cross-seed p95 TV <= 0.35
```

No gate is relaxed.

## Completed 5×64 comparison matrix

All rows below are 5 CFR iterations × 64 roots = 320 roots/seed.

| candidate | mean TV | p95 TV | relative interpretation |
|---|---:|---:|---|
| size4 no damping | `0.266591` | `0.567002` | authoritative durability baseline |
| size4 decay tremble e15 | `0.231886` | `0.475154` | material improvement |
| size4 decay tremble e30 | `0.217853` | `0.457102` | stronger |
| size4 decay tremble e45 | `0.211607` | `0.448567` | strongest uniform-tremble row |
| size4 temporal blend w75 | `0.222885` | `0.481673` | material improvement |
| **size4 temporal blend w50** | **`0.179915`** | **`0.395478`** | **best completed durable row** |
| size4 first-transition-only e30 | `0.239409` | `0.512631` | helps, but repeated stabilization matters |
| size1 no damping | `0.438845` | `0.878729` | poor |
| size1 decay tremble e30 | `0.395333` | `0.798287` | damping helps independently but insufficient |
| Direct Behavior control | `0.276185` | `0.828670` | closed as durable solution |
| Direct Behavior aggregated regret | `0.307350` | `0.914166` | closed; worse than control |

The strongest completed result, temporal w50, improves the size4 baseline by approximately:

```text
mean TV: 32.5%
p95 TV:  30.3%
```

but still misses the frozen cross-seed gates by:

```text
mean gap = 0.029915
p95 gap  = 0.045478
```

This is the strongest evidence so far that **iteration-to-iteration policy replacement itself is a major source of instability**. The w50 result is substantially stronger than w75, and first-transition-only damping is weaker than continued temporal stabilization, indicating that the problem is repeatedly regenerated rather than seeded only once.

## Size8 — short-horizon gate pass, durability pending

Workflow `31440425854`, evidence commit `bfe6d4845600c3eafed36c85c0113756763f6910`:

```text
2×128
mean TV = 0.1396147311  PASS
p95 TV  = 0.3296890855  PASS
fits    = PASS
```

This is the first current Generation-2 short-horizon candidate to clear both frozen cross-seed gates. It is not accepted as durable because the size4 history already demonstrated strong two-iteration success followed by five-iteration decay.

Its mandatory 5×64 run is workflow `31446308103`. Latest physical state: build/regression PASS, smoke PASS, physical five-iteration durability running.

## Highest-priority composition — size8 + temporal w50

Two mechanisms are independently supported and target different failure modes:

1. increasing Advantage policy-mixture size reduces approximation/sign disagreement at a fixed feedback state;
2. temporal blending reduces feedback-depth amplification across iterations.

Commit `8166871908f0580cd3170ebd038ca0ad83072951` therefore added the direct composition:

```text
partial-exact opponent expectation level 2
Advantage policy-mixture size 8
policy used for feedback = 0.50 * current + 0.50 * previous iteration
first feedback reference = exact zero-regret uniform policy
5 iterations × 64 roots
320 roots/seed
```

Workflow `31448623827` is active. At the latest physical poll, build/regression had passed and the size8 temporal-w50 smoke was running. The physical 5×64 phase follows automatically after smoke.

This composition is deliberately **not** an assumption of additivity: it must demonstrate its own full five-iteration result. It is nevertheless the highest-value experiment because it combines the strongest short-horizon and strongest durable mechanisms without changing the frozen gates or the authoritative deck schedule.

## Remaining base-matrix experiments

### Regret-floor policy mixture

Workflow `31444922236`, epsilon `.05 / .10`, physically running both 5×64 jobs after build and smoke PASS.

This attacks the empirically fragile near-zero positive-regret boundary before member policies are averaged.

### Uncertainty-adaptive damping

Workflow `31442367579`, disagreement scale `.50 / 1.00`, cap `.50`, physically running both 5×64 jobs after build and smoke PASS.

This asks whether damping should be concentrated only where ensemble members disagree rather than applied globally.

## Causal conclusions from completed branches

### Global damping has an independent signal

Size1 at 5×64:

```text
no damping:       mean 0.438845 / p95 0.878729
epsilon0=.30:     mean 0.395333 / p95 0.798287
```

Thus damping itself helps even without ensembling, but ensembling is the larger effect.

### Continued temporal inertia is stronger than one-shot damping

```text
first-transition-only e30: mean 0.239409 / p95 0.512631
temporal w50:              mean 0.179915 / p95 0.395478
```

The instability is therefore not adequately explained as a one-time bad first transition. Later fitted-policy replacement remains material.

### Direct Behavior is closed as a durable solution

Ordinary Direct Behavior degraded to `0.276185 / 0.828670` at five iterations. The cleaner aggregated-regret order degraded further to `0.307350 / 0.914166`. The smooth surrogate remains useful as a causal control but is neither durable nor production-equivalent.

### Final AveragePolicy ensemble remains a downstream residual layer

At 2×128 after size4 policy-mixture CFR:

```text
final ensemble size1: mean 0.179750 / p95 0.434644
size2:                mean 0.159165 / p95 0.404792
size4:                mean 0.138377 / p95 0.368730
```

This layer is reserved for a durable upstream winner. It is not assumed to preserve the same gain after five iterations; that must be tested if needed.

## Automatic evidence control

The base consolidator is `SPINCORE_R7_3_DURABILITY_MATRIX_SUMMARY_V4` and expects 15 candidate rows plus the authoritative baseline. At the latest audit:

```text
completed base candidate rows = 10
pending base candidate rows   = 5
```

The pending rows are size8 no-damping, regret-floor e05/e10 and uncertainty damping s05/s10. The new size8+temporal-w50 composition is supplemental and intentionally outside the frozen base matrix.

## Promotion rule

No candidate moves to 640 merely because it is best among the experiments. Promotion requires:

1. all frozen per-seed fit gates PASS;
2. material improvement in **both** mean and p95 versus `0.2665907145 / 0.5670017600` at 5×64;
3. preferably full cross-seed gate clearance at the five-iteration horizon; any residual miss must be explicitly solved and retested rather than waived;
4. fresh-process reproducibility;
5. the smallest/interpretable mechanism among statistically comparable candidates;
6. explicit freeze/versioning of any changed behavior semantics;
7. deterministic continuous-vs-stop/restore/continue checkpoint recertification;
8. frozen gates unchanged.

If size8+temporal-w50 clears or narrowly approaches both gates, the immediate next stage is **semantic freeze, fresh-process reproducibility and checkpoint/resume recertification**. Acceptance-scale 640 comes only after those checks.
