# R7.3 regret-floor policy-mixture diagnostic — 2026-08-10

`READY FOR TABLES = NO`. Frozen R7.3 gates are unchanged.

## Motivation

The generation-2 causal evidence localizes the first material divergence to the fitted Advantage behavior feedback boundary. Independent Advantage fits on the same frozen memory disagree materially after the production hard regret-matching map even when their normalized regression error passes the frozen fit gate.

The same-memory sign-sensitivity evidence (`validation/R7_3_ADVANTAGE_FIT_SIGN_SENSITIVITY_256.json`) reports:

```text
average hard-RM pairwise mean TV = 0.2243488008
average hard-RM pairwise p95 TV  = 0.7575294160
positive-regret support equal     = 0.5548502604
```

A substantial fraction of TV mass is concentrated where at least one legal predicted regret lies close to zero. Post-hoc scale-relative epsilon floors reduced pairwise disagreement; epsilon `0.05` typically reduced mean TV about 10%, while epsilon `0.10` reduced it about 18% in the frozen same-memory probe.

This directly motivates testing the mapping itself rather than another global behavior surrogate.

## Candidate mapping

For every independently fitted Advantage member and state, let

```text
scale = RMS(predicted legal advantages)
floor = epsilon * max(scale, 1e-8)
weight(a) = max(advantage(a), 0) + floor
policy_member(a) = weight(a) / sum_legal(weight)
```

Then the four member policies are averaged exactly as in the existing policy-mixture ensemble.

Two values are tested:

```text
epsilon = 0.05
epsilon = 0.10
```

The floor is scale-relative, so multiplying all predicted advantages by a positive scalar does not materially change the resulting policy. At epsilon zero, the construction reduces to ordinary hard regret matching (including legal-uniform fallback when every regret is non-positive).

## What is preserved

The diagnostic does **not** change:

- partial-exact opponent estimator level 2;
- Advantage memory construction or LCFR sample weights;
- neural architecture or member fitting;
- authoritative deal formula;
- recovered primary coupled RNG stream;
- AveragePolicy training;
- frozen R7.3 gates.

It changes only the mapping from fitted Advantage values to the behavior policy used for subsequent CFR collection.

## Why this differs from global tremble

Uniform tremble mixes every state toward uniform by a time-based epsilon schedule. The regret floor acts locally in predicted-regret scale and directly regularizes the discontinuous sign/support boundary that the same-memory forensic identified.

This makes it complementary to the active uncertainty-adaptive damping experiment:

- **global tremble**: regularize all states according to iteration;
- **uncertainty damping**: regularize states according to inter-member policy disagreement;
- **regret floor**: regularize each member according to its local regret magnitude near the hard-RM boundary.

The five-iteration durability matrix will determine which intervention best improves both center and tail.

## Physical experiment

Workflow `31444922236` runs two independent jobs at the actual failing durability horizon:

```text
ensemble size = 4
partial-exact opponent levels = 2
iterations = 5
roots/iteration = 64
roots/seed = 320
regret-floor epsilon = 0.05 / 0.10
```

Both jobs must pass the existing frozen Advantage and final AveragePolicy fit gates. Success is measured against the authoritative no-floor size-4 durability baseline:

```text
mean TV = 0.2665907145
p95 TV  = 0.5670017600
```

A candidate must materially improve **both** metrics. Merely becoming the best row in the matrix is insufficient for promotion.

## Promotion boundary

Even a five-iteration winner remains experimental. Before any 640 acceptance-scale run, the exact mapping and epsilon must be versioned, fresh-run reproducibility must pass, and continuous-vs-stop/restore/continue checkpoint determinism must be recertified. Strategic impact must also be audited because adding positive mass to previously non-positive regret actions changes behavior semantics.

No gate relaxation is permitted.
