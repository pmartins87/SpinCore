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

`global_root` is continuous across CFR iterations. The recovered acceptance path preserves one persistent live `bundle.batch_rng` through collection and primary training in execution order unless a diagnostic explicitly declares otherwise.

Authoritative 640 references remain failures:

| candidate | mean TV | p95 TV | fit gates |
|---|---:|---:|---|
| corrected 640 | `0.477649` | `0.902403` | PASS |
| strong-Advantage 640 | `0.464474` | `0.886204` | PASS |
| partial-exact level 2 | `0.436110` | `0.888244` | PASS |
| partial-exact level 2 strong-fit | `0.412893` | `0.871708` | PASS |

## Causal transition — confirmed

Under the current authoritative partial-exact contract, iteration 1 shared strategy targets are identical:

```text
shared-target weighted mean TV = 0.0
shared-target p95 TV           = 0.0
```

After the first fitted Advantage behavior feeds back into collection, iteration 2 becomes:

```text
shared-target weighted mean TV = 0.473946
shared-target p95 TV           = 1.0
```

Therefore the dominant known transition remains:

```text
Advantage approximation -> regret mapping -> behavior -> sampled trajectories -> next strategy targets
```

The final AveragePolicy is downstream of an already-divergent target distribution.

## Policy-mixture size 4 — strong short-horizon signal, failed durability test

At 2 iterations × 128 roots, paired partial-exact level 2 plus four independently fitted Advantage members, each hard-regret-matched before probability averaging, gave:

```text
mean TV = 0.171940
p95 TV  = 0.413605
fit gates = PASS
```

This was a ~30% mean and ~34% p95 reduction versus the paired size-1 control (`0.245656 / 0.628706`).

The mandatory longer-feedback test has now completed: workflow `31432403037`, 5 iterations × 64 roots = 320 roots/seed.

```text
mean TV = 0.266591
p50 TV  = 0.246805
p95 TV  = 0.567002
max TV  = 0.905547
fit gates = PASS
```

The two-iteration gain therefore **decays materially under five CFR feedback cycles**. This is a decisive negative result for immediate size-4 acceptance escalation.

**Decision: do not launch the prepared size-4 640 candidate.** More roots would not answer the feedback-depth failure.

## Direct Behavior — useful causal clue, not a production algorithm

Authoritative 256 E2E:

```text
mean TV = 0.142553   PASS
p95 TV  = 0.426860   FAIL
```

The surrogate itself underfits the sample-level regret-matched targets well above the reference `0.12` TV threshold. Because regret matching is nonlinear, this is not theoretically equivalent to recovered Deep CFR. The result is retained only as evidence that **explicit smoothing around the first feedback transition can materially reduce instability**.

## Current residual-tail program

### 1. Policy-mixture size 8

The first size-8 smoke attempt correctly failed before physical evidence because the shared paired runner only allowed sizes 1/2/4. That compatibility limit was explicitly extended to 8; no failed-run fallback JSON is treated as evidence.

Corrected workflow `31440425854` is active. Its question is narrow: does doubling independent regret-policy members continue to reduce the two-iteration p95, or has the ensemble already saturated?

Any size-8 success still requires a five-iteration compounding screen before 640.

### 2. Policy-mixture + final AveragePolicy ensemble

Workflow `31440493410` is active. It keeps size-4 policy-mixture behavior during CFR, then trains final AveragePolicy ensembles of sizes 1/2/4 on the frozen strategy memory without perturbing the primary RNG stream.

This is a factorial residual-tail test. Prior final-policy ensembling alone reduced p95 only ~4.7%, so it will be promoted only if the combination is materially stronger than that.

### 3. Support-conditioned residual-tail forensic

A new authoritative 256 diagnostic separates final-policy disagreement on:

- seed-A support;
- seed-B support;
- exact byte-identical shared SPNNIV1 observations;
- exact one-sided observations unique to A or B.

This determines how much of the remaining p95 is true shared-state disagreement versus off-support generalization/extrapolation after upstream policy-mixture stabilization.

### 4. Robust policy aggregation

A same-memory eight-replica diagnostic compares ordinary probability averaging with coordinatewise median and trimmed-mean aggregation **after each member has already been hard-regret-matched**.

This directly tests whether rare ensemble-member outliers dominate the p95 tail. Only a large same-memory tail reduction earns an E2E mapping test.

### 5. Independent five-iteration reproducibility run

A parser-compatibility edit unintentionally triggered a second physical copy of the 320 compounding workflow. Rather than discard it, it is retained as a determinism check. Because the edit changes only allowed parser choices and not size-4 semantics, its numerical output should reproduce the completed `0.266591 / 0.567002` result if the physical pipeline is deterministic across fresh runners.

## Closed / deprioritized primary branches

- raw root scaling beyond 1280;
- independent x8/x16 path multiplication;
- common-path RNG;
- antithetic x4;
- exhaustive opponent expectation;
- merely raising Advantage optimizer ceiling;
- behavior-aware MSE auxiliary objective;
- exact weighted duplicate-target aggregation;
- behavior-aware multistart model selection;
- raw Advantage ensemble sizes 2/4 as standalone solution;
- final AveragePolicy ensemble as standalone solution;
- legal-action common-mode centering;
- card/suit representation rewrite as dominant explanation;
- policy-mixture size 4 direct escalation to 640.

## Immediate decision tree

1. Resolve size 8, robust aggregation, support-conditioned tail decomposition, combined final-policy ensemble, and the independent 320 reproducibility run.
2. If size 8 or robust aggregation gives a materially better p95 at two iterations, require a five-iteration compounding screen before 640.
3. If all static ensemble variants still decay with feedback depth, move to an **explicit versioned temporal behavior damping/interpolation** diagnostic around the first Advantage-fit transition. This is the next algorithmic lever already justified by the Direct Behavior result.
4. Any new behavior semantics must be explicitly versioned and must pass deterministic continuous-vs-stop/restore/continue recertification before acceptance-scale promotion.
5. No gate relaxation.

Historical pre-loss `0.3714 / 0.6878` remains historical evidence only and is never substituted for generation-2 gates.

## Recovery invariants

- `TRUE_HEADS_UP` and `THREE_HANDED` remain separate whole-hand domains.
- Production utility remains exact explicit-payout ICM continuation delta.
- Ambiguous equal-stack simultaneous elimination with unequal unresolved payouts fails closed.
- Every meaningful recovery/evolution step is persisted directly to `main`.

`READY FOR TABLES = NO`.
