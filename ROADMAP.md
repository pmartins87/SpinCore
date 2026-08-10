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

The mandatory longer-feedback test completed in workflow `31432403037`, 5 iterations × 64 roots = 320 roots/seed:

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

The surrogate itself underfits the sample-level regret-matched targets well above the reference `0.12` TV threshold. Because regret matching is nonlinear, this is not theoretically equivalent to recovered Deep CFR. The result is retained as causal evidence that **explicit smoothing around the first feedback transition can materially reduce instability**.

## Parallel residual-tail program

### A. Static ensemble tail tests

1. **Policy-mixture size 8** — corrected workflow `31440425854`; build/regression and smoke PASS, physical 256-root candidate running. It asks whether doubling the number of regret-policy members pushes the two-iteration tail lower or whether size 4 has saturated. Any success must still survive five iterations before 640.
2. **Policy-mixture + final AveragePolicy ensemble** — workflow `31440493410`; physical 256-root factorial running after smoke PASS. Final policy sizes 1/2/4 are evaluated on the same size-4 policy-mixture CFR memory.
3. **Robust probability aggregation** — workflow `31440742014`; same-memory physical screen running after smoke PASS. It compares ordinary mean, coordinatewise median and trimmed mean after per-member hard regret matching. Only a large p95 reduction earns E2E testing.
4. **Support-conditioned tail forensic** — workflow `31440576227`; physical 256-root run active. It separates exact shared SPNNIV1 observations from one-sided/off-support observations after policy-mixture CFR.

### B. Five-iteration temporal damping program

Because size-4 policy mixture improves two iterations but decays over five, temporal damping is tested **directly at the failing 5×64 horizon**, not on another short proxy.

#### Decaying uniform tremble — workflow `31441018067`

Three size-4 candidates run in parallel, all after smoke PASS:

```text
epsilon0 = 0.15, 0.30, 0.45
decay    = 0.50 per fitted iteration
```

The behavior used for subsequent collection is:

```text
pi_used = (1 - epsilon_k) * pi_policy_mixture + epsilon_k * uniform
```

with a geometrically vanishing intervention. This tests whether explicit early-feedback regularization can prevent divergence without permanently flattening the policy.

#### Size-1 tremble factorial — workflow `31441110526`

Two 5×64 candidates run after smoke PASS:

```text
ensemble size 1, epsilon0 = 0.00
ensemble size 1, epsilon0 = 0.30
```

Together with the completed size-4/no-tremble baseline and size-4/epsilon0=0.30 candidate, this yields a causal 2×2 comparison of **ensemble × temporal damping** rather than attributing any improvement to both at once.

#### Temporal previous-policy blending — workflow `31441224117`

Two size-4 5×64 candidates run after smoke PASS:

```text
current policy weight = 0.50
current policy weight = 0.75
```

At the first feedback transition, the reference policy is exact zero-regret uniform. At later transitions, current policy mixture is blended with the **previous iteration's fitted policy mixture**. This directly tests whether abrupt iteration-to-iteration behavior replacement is the mechanism behind the compounding failure.

All temporal damping/blending variants are explicitly algorithmic diagnostics. They are **not** declared equivalent to recovered Deep CFR and cannot become production semantics without versioning, strategic audit, and deterministic checkpoint/resume recertification.

### C. Physical reproducibility controls

Parser compatibility changes unintentionally triggered fresh copies of the existing size-4 256 and size-4 320 workflows. They are retained as useful deterministic fresh-run controls:

- size-4 paired 256 reproduction: workflow `31440366942`, expected `0.171940 / 0.413605`;
- size-4 320 reproduction: workflow `31440366909`, expected `0.266591 / 0.567002`.

A mismatch beyond numerical roundoff is itself a blocker requiring nondeterminism investigation before any checkpoint/RNG semantic work.

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

1. No 640 escalation unless a candidate first beats the five-iteration durability baseline `0.266591 / 0.567002` with every frozen fit gate PASS.
2. Resolve size 8, robust aggregation, combined final-policy ensemble and support-conditioned tail forensic to determine whether a better *static* mapping exists.
3. Resolve the five-iteration tremble dose-response, size-1 factorial and temporal previous-policy blend. Prefer the **smallest and most interpretable** mechanism that improves both mean and p95 durably.
4. If a static size-8/robust mapping wins at two iterations, it still requires its own five-iteration compounding gate before 640.
5. If temporal damping wins, first freeze/version its exact schedule and add continuous-vs-stop/restore/continue determinism before acceptance-scale testing.
6. No gate relaxation.

Historical pre-loss `0.3714 / 0.6878` remains historical evidence only and is never substituted for generation-2 gates.

## Recovery invariants

- `TRUE_HEADS_UP` and `THREE_HANDED` remain separate whole-hand domains.
- Production utility remains exact explicit-payout ICM continuation delta.
- Ambiguous equal-stack simultaneous elimination with unequal unresolved payouts fails closed.
- Every meaningful recovery/evolution step is persisted directly to `main`.

`READY FOR TABLES = NO`.
