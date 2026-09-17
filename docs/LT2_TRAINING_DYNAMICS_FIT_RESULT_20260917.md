# SpinCore — LT2 training-dynamics fit audit result

Date: 2026-09-17
Status: **FIT BURDEN IS LARGE — BUDGET SUFFICIENCY NOT YET IDENTIFIED — CONTROLLED HELD-OUT BUDGET SWEEP NEXT**

## Source

Read-only audit of preserved checkpoints:

- Stage A: iteration 3000 / 1.8M roots, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`;
- Stage B: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

The audit sampled 25,000 items from each Advantage and AveragePolicy reservoir, separately for 3H and HU. No optimizer step or checkpoint mutation occurred.

## Observed fit metrics

Weighted metrics:

| checkpoint/domain | Advantage error removed vs zero | Advantage policy TV | Advantage argmax | Policy gap closed vs uniform | Policy KL | Policy TV | Policy argmax |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage A 3H | 9.66% | 0.6183 | 26.36% | 12.81% | 0.6381 | 0.4307 | 50.32% |
| Stage A HU | 14.79% | 0.6311 | 21.63% | 18.76% | 0.6259 | 0.4248 | 52.04% |
| Stage B 3H | 9.98% | 0.5969 | 33.56% | 12.98% | 0.6323 | 0.4286 | 51.46% |
| Stage B HU | 14.17% | 0.6058 | 26.40% | 16.27% | 0.6387 | 0.4318 | 49.20% |

The stored networks therefore remain far from the individual stored targets under these diagnostics. Stage B HU AveragePolicy is modestly worse than Stage A HU on KL, TV, argmax and fraction of the uniform-to-target gap closed. Stage B Advantage fit is broadly similar to Stage A rather than showing a new catastrophic regime.

## Important interpretation limit

These absolute fit metrics do **not** by themselves prove optimizer underfitting.

Advantage targets are Monte-Carlo/external-sampling targets and contain conditional variance. AveragePolicy memory contains behavior-policy targets from many historical iterations; an individual historical target can differ from the conditional average the network should approximate. Therefore some per-sample MSE/KL/TV is irreducible even for a well-optimized finite model.

The audit does establish that approximation error/burden is large enough to require a controlled optimizer-budget experiment before blaming root count, representation, network capacity, or target semantics.

## Mechanics clarified

The current Advantage network is reset from scratch every iteration and receives exactly 100 optimizer steps with batch size 1024. At Stage B capacity, that is 102,400 sample draws against a 2,000,000-item reservoir. Under a simple independent-draw coverage approximation, this corresponds to only about 99.8k unique items, roughly 5.0% of the reservoir, seen by the freshly reset network in that iteration. This is a scale diagnostic, not a pass threshold.

The AveragePolicy wording also needs precision: `policy_steps=4000` means 4000 optimizer steps **per milestone finalization**, not 4000 cumulative lifetime steps. The Stage-B counter is 12,000 cumulative policy optimizer steps; Stage A was 8,000. The current question is therefore whether an additional finalization budget materially improves held-out fit, not whether the policy has only ever seen 4000 steps.

## Next bounded causal experiment

Run `tools/run_lt2_stage_b_fit_budget_sweep.sh`.

It uses Stage B only, does not collect roots, and never saves over the source checkpoint. For each 3H/HU memory it creates a deterministic 25k held-out set and excludes those exact items from optimization sampling.

Advantage sweep:

- fresh deterministic reset;
- cumulative budgets `0, 25, 50, 100, 200, 400, 800, 1600` steps;
- canonical 100-step budget is explicitly on the curve;
- evaluate held-out MSE, zero-baseline fraction removed, induced-policy TV and argmax at each point.

AveragePolicy sweep:

- start from the stored Stage-B policy+optimizer state;
- cumulative **additional** budgets `0, 1000, 2000, 4000, 8000` steps;
- canonical finalization increment (4000) is explicitly on the curve;
- evaluate held-out CE/KL/TV/argmax and uniform-gap fraction at each point.

The geometric budgets are diagnostic, not pass scores. The experiment asks whether held-out fit keeps improving materially beyond the production budget and where the curve begins to flatten.

## Decision after sweep

- if Advantage held-out fit improves strongly beyond 100, test a changed Advantage budget in a small isolated training continuation;
- if AveragePolicy held-out fit improves strongly with additional steps, produce an isolated refit candidate before new roots and benchmark it against the same weak baselines;
- if either curve plateaus early while fit remains poor, optimizer-step count is not the main bottleneck; move to model capacity/representation/target-variance analysis;
- if both curves plateau early, inspect HU target generation, reservoir weighting and state/action concentration before any new long-root run.

No long training beyond iteration 7500 is admitted yet. DeepCrusher remains deferred.