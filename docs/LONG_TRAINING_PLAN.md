# SpinCore — Long-Training Plan

Status: **LT2 STAGE B PASS — ROOT TRAINING PAUSED — BROAD FAI CALIBRATION DOES NOT EXPLAIN RESOLVED FIRST-DIVERGENCE LOSS — FULL-POPULATION RECONCILIATION ACTIVE**
Date: 2026-09-19

## Preserved milestones

Stage A:
- iteration 3000 / 1.8M roots;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Never rewrite either checkpoint.

## Current scientific conclusion

The Jammer current-behavior regression is real in the paired weak-baseline evaluation:

- total B-A `-8.4304`, resolved;
- FAI first-divergence contribution `-6.2267`, resolved;
- FAI explains 73.86%.

But the broad 48-anchor FAI calibration does not show Stage B globally worse.

### Broad expected policy regret

Stage A:
- `32.885` chips.

Stage B:
- `20.984`.

B-A:
- `-11.901`, seed-cluster CI crosses zero.

### Canonical action-gap MSE

B-A:
- `-0.00058635`;
- resolved Stage-B improvement.

### FOLD-vs-CONTINUE class error

B-A:
- `-0.08432`;
- resolved Stage-B improvement.

Thus the simple RM-calibration hypothesis is not supported on the broad 48-anchor sample.

## Metric correction

Do not use strict positive-support agreement as a causal training objective from this audit.

Because the stage-specific true Advantage is centered on the same `sigma` produced by the raw model, a pure optimal action can have true Advantage exactly zero while requiring a positive raw output to be selected by production regret matching.

Policy regret and canonical action gaps remain valid.

## Required reconciliation

Before any intervention, compare the two results on the same full natural evaluation population.

Canonical contract:

`docs/LT2_JAMMER_FAI_POPULATION_RECONCILIATION_20260919.md`.

For every Jammer HU seat-run:
- replay paired A/B current behavior;
- if a common FAI state is reached before earlier divergence, inspect it before hero action;
- clone exact dealt solver state for every legal action;
- apply action to terminal;
- read hero chip delta;
- compute deterministic expected Stage-B-minus-A policy value.

Also use the exact previous RNG to reproduce the sampled FAI first-divergence contribution.

Hard gate:
- sampled contribution must reproduce `-6.22672064777328` exactly.

## Why actual-deal Q is allowed here

Per-state `Q_actual` uses hidden opponent cards and the full future board, so it is not a deployable infoset target.

Across every natural evaluation deal, however, it is an unbiased counterfactual estimator under the exact same distribution as the weak-baseline test.

Its purpose is attribution/reconciliation only.

## Decision logic

If deterministic expected full-population FAI contribution is resolved negative:
- diagnose the harmful policy-mass shift regime on that same population;
- only then use high-budget infoset references on a pre-registered subset.

If it is neutral/positive:
- the stochastic first-divergence attribution requires methodological correction.

No K4 training, no RM loss modification, and no resumed root training before this gate.

## Immediate direction

1. Keep Stage A/B frozen.
2. Run `bash tools/run_lt2_jammer_fai_population_reconciliation.sh`.
3. Wait for `LT2_JAMMER_FAI_POPULATION_RECONCILIATION_PASS`.
4. Send `SpinCore_LT2_jammer_fai_population_reconciliation.json`.
5. Keep holdout `20261001..20261006` untouched.
6. Do not train.
