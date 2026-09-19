# SpinCore — LT2 Jammer FAI full-population counterfactual reconciliation

Date: 2026-09-19
Status: **ACTIVE — RECONCILE RESOLVED FIRST-DIVERGENCE LOSS WITH NON-DEGRADING 48-ANCHOR BROAD CALIBRATION**

## Trigger

Two diagnostics now disagree in appearance:

### Full paired first divergence

Jammer current behavior:
- total B-A `-8.4304`, resolved;
- FAI additive contribution `-6.2267`, resolved.

### 48-anchor broad low-noise FAI calibration

Stage B does **not** look broadly worse:
- policy-regret B-A `-11.901` chips, CI crosses zero;
- canonical action-gap MSE improves significantly;
- FOLD-vs-CONTINUE class-error mass improves significantly.

Therefore do not design an RM-specific intervention yet.

The contradiction must be reconciled on the exact natural evaluation population.

## Design

Use all forensic seeds:

`20260920..20260925`.

Use all 5000 scenarios/seed.

HU only.

For each Jammer hero seat-run:

1. replay Stage A and Stage B current behavior with identical scenario, deal and RNG;
2. stop if an earlier hero action diverges;
3. if both reach the same preflop FAI state:
   - record the state before the hero action;
   - compute Stage-A and Stage-B current behavior distributions;
   - clone the exact solver state once per legal action;
   - apply each legal action;
   - require terminal state;
   - read hero terminal chip delta.

This gives `Q_actual[a]` for the exact hidden hand and full board already dealt in that evaluation scenario.

## Important interpretation

`Q_actual` is **not** an infoset target for one state because it uses hidden opponent cards and the full future board.

But over the complete natural evaluation population, it is an unbiased counterfactual estimate under exactly the same deal distribution used by the weak-baseline benchmark.

This audit is therefore a reconciliation / attribution tool, not a training target.

## Deterministic expected FAI contribution

For every common FAI state compute:

`DeltaV = sum_a (sigma_B[a] - sigma_A[a]) Q_actual[a]`.

For seat-runs that do not reach a common FAI state before earlier divergence, contribution is zero.

Cluster by scenario exactly as the previous first-divergence audit.

## Paired-sampling reproduction contract

At the same FAI state, also sample Stage A and Stage B actions using the exact hero RNG stream from the previous forensic.

The resulting sampled additive FAI contribution must reproduce:

`-6.22672064777328`

to numerical identity.

If it does not, the reconciliation audit fails and no interpretation is allowed.

## Additional decomposition

Report:

- common-FAI reach rate;
- FAI sampled-divergence rate;
- Stage-A and Stage-B fold mass;
- fold-mass B-A;
- policy TV;
- non-FOLD actual-Q spread;
- additive expected contribution by:
  - B_MORE_FOLD;
  - B_LESS_FOLD;
  - NO_FOLD_SHIFT;
- legal-action signature;
- common public-action path length.

## Decision logic

### If deterministic expected FAI contribution is resolved negative

The first-divergence result is genuine at the full-population policy-value level.

Then use the full population to identify which fold-mass shift regime carries the loss, and only afterward run low-noise infoset references on a pre-registered state subset.

### If deterministic expected FAI contribution is neutral/positive

Then the previously resolved sampled first-divergence loss is not reflecting the expected policy-value difference and the coupling / attribution method requires correction.

### If sampled contribution fails to reproduce prior exactly

Stop immediately and debug methodology.

## Launcher

```bash
bash tools/run_lt2_jammer_fai_population_reconciliation.sh
```

Expected marker:

`LT2_JAMMER_FAI_POPULATION_RECONCILIATION_PASS`.

Holdout `20261001..20261006` remains untouched.

No training is authorized.
