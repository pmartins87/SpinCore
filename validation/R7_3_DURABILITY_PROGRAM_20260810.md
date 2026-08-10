# R7.3 five-iteration durability program — 2026-08-10

`READY FOR TABLES = NO`. Frozen R7.3 gates remain unchanged.

## Why the optimization target changed

The authoritative paired size-4 Advantage policy-mixture candidate was highly effective over two CFR iterations (`mean TV 0.171940`, `p95 0.413605` versus paired size-1 `0.245656 / 0.628706`). However the mandatory 5×64 compounding run, workflow `31432403037`, completed at:

```text
mean TV = 0.2665907145
p50 TV  = 0.2468046695
p95 TV  = 0.5670017600
max TV  = 0.9055466652
fit gates = PASS
```

Therefore short-horizon variance reduction is no longer sufficient evidence for promotion. The next mechanism must demonstrate **feedback-depth durability** before any 640-root acceptance escalation.

This change is consistent with the independent causal forensic: exact shared strategy targets are identical in iteration 1 and become `mean TV 0.473946 / p95 1.0` immediately after the first fitted Advantage behavior feeds back into collection.

## Five-iteration comparison baseline

The primary no-damping size-4 baseline is:

```text
partial-exact level 2
Advantage policy-mixture size 4
5 CFR iterations × 64 roots
320 roots/seed
mean/p95 = 0.2665907145 / 0.5670017600
```

All new temporal mechanisms must be compared against this horizon, not against the easier two-iteration result.

A fresh duplicate physical run (`31440366909`) is retained as a determinism check and should reproduce this output up to numerical roundoff.

## Experiment family A — static residual-tail controls

These remain useful for explaining the two-iteration tail, but they do not earn 640 promotion without their own five-iteration test.

- `31440425854`: policy-mixture size 8, authoritative paired 256.
- `31440493410`: size-4 policy-mixture CFR plus final AveragePolicy ensembles 1/2/4.
- `31440576227`: support-conditioned final-policy disagreement on A support, B support, exact shared observations and one-sided observations.
- `31440742014`: same-memory robust policy aggregation.

The robust aggregation run is complete. Ordinary probability averaging remains superior:

```text
mean aggregation:    mean 0.137421 / p95 0.388113
median aggregation:  mean 0.143639 / p95 0.488827
trimmed mean:        mean 0.143639 / p95 0.488827
```

Diagnosis `ROBUST_POLICY_AGGREGATION_NOT_MATERIAL`. Rare single-member outliers are therefore not the main explanation of the size-4 policy-mixture tail; coordinatewise robust aggregation actually worsens it.

## Experiment family B — decaying uniform tremble

Workflow `31441018067` runs three size-4 candidates at the full failing horizon:

```text
epsilon0 = 0.15 / 0.30 / 0.45
epsilon_k = epsilon0 * 0.5^(k-1)
```

For each fitted behavior:

```text
pi_used = (1 - epsilon_k) * pi_policy_mixture + epsilon_k * uniform_legal
```

This is deliberately a vanishing intervention: the earliest unstable feedback transition is damped most strongly and later behavior approaches the regret-policy mixture.

## Experiment family C — 2×2 ensemble × tremble causal factorial

Workflow `31441110526` provides the missing size-1 controls at the same 5×64 horizon:

```text
size 1, epsilon0 = 0.00
size 1, epsilon0 = 0.30
```

Together with size-4/no-tremble and size-4/epsilon0=0.30, this separates the effect of ensembling from the effect of explicit temporal damping.

## Experiment family D — previous-policy temporal blending

Workflow `31441224117` tests two size-4 candidates:

```text
current policy weight = 0.50 / 0.75
```

At the first feedback transition the reference is the exact zero-regret uniform policy. At later transitions:

```text
pi_used = w * pi_current + (1-w) * pi_previous_iteration
```

This directly targets abrupt iteration-to-iteration replacement of a noisy fitted regret policy rather than injecting permanent exploration.

## Experiment family E — first-transition-only intervention

Workflow `31441567261` uses:

```text
epsilon0 = 0.30
decay = 0.0
```

so only the first fitted-Advantage feedback transition is damped. This is a high-value causal discriminator: if it performs like the decaying schedule, most later instability is seeded by the first break; if it fails while decaying damping succeeds, repeated stabilization is required.

## Experiment family F — Direct Behavior durability control

Workflow `31441650915` repeats the experimental Direct Behavior surrogate at 5×64. The two-iteration run passed the frozen mean gate (`0.142553`) but failed p95 (`0.426860`) while its surrogate itself underfit the reference regret-matched targets.

This five-iteration run is **not** a production candidate. It asks whether the broad smoothing effect that helped at two iterations itself survives compounding. The answer informs temporal-damping design even though the surrogate remains algorithmically non-equivalent.

## Promotion rule

No candidate moves to 640 merely because it is best among the experiments. Promotion requires all of the following:

1. all frozen per-seed fit gates PASS;
2. material improvement in **both** mean and p95 versus `0.2665907145 / 0.5670017600` at 5×64;
3. no evidence that the gain is a non-reproducible fresh-run artifact;
4. the smallest/interpretable mechanism wins when statistically comparable;
5. any new behavior semantics are explicitly versioned;
6. deterministic continuous-vs-stop/restore/continue checkpoint recertification precedes acceptance-scale use;
7. frozen gates are not changed.

A mechanism that only improves the two-iteration screen remains diagnostic evidence, not an R7.3 solution.
