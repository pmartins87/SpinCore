# R7.3 exact reproducibility — deferred certification decision

Date: 2026-08-12

## Decision

The R7.3 exact fresh-process reproducibility requirement remains **unsatisfied** and is **not** reclassified as PASS.

It is no longer a development blocker for R7.4 and later engineering work, provided the frozen candidate independently passes the unchanged strategy-quality gates at the 5 x 128 = 640 roots/seed acceptance scale.

Exact reproducibility is now tracked as certification debt and MUST be closed before `READY FOR TABLES` can become true. No later stage may erase, overwrite, or silently reinterpret this exception.

## Why the sequencing changes

The failed recertification proves that the same frozen source/seed procedure is not numerically identical in the current runtime. The latest persisted fresh-process report records:

- `difference_count = 734` under the unchanged `1e-9` comparison tolerance;
- historical cross-seed policy mean TV = `0.1329178512096405`;
- fresh cross-seed policy mean TV = `0.14696117863059044`;
- frozen mean-TV gate = `0.15`;
- historical cross-seed p95 TV = `0.2854667007923126`;
- fresh cross-seed p95 TV = `0.33013119905225885`;
- frozen p95-TV gate = `0.35`.

Therefore this is not merely bit-level floating-point dust: the training trajectory changes enough to move aggregate policy metrics. At the same time, the observed fresh aggregate metrics remain inside the precommitted strategy-quality limits. This makes exact reproducibility primarily a certification/provenance problem at the present development stage, not proof that the candidate strategy is strategically invalid.

The mean-TV headroom in the fresh run is about `0.0030388214` and the p95 headroom is about `0.0198688009`, so the continuation must not rely on those aggregate observations alone. A new physical 640-root strategy bridge must re-run the frozen candidate and enforce the original quality gates.

## Provisional 640 strategy bridge

R7.4 may begin provisionally only if a fresh exact-source 5 x 128 run satisfies all of the following without changing thresholds:

- exact frozen winner source and semantic freeze are used;
- iterations = 5;
- roots/iteration = 128;
- roots/seed = 640;
- exact opponent levels = 2;
- deck formula remains `seed*1000003 + global_root*97 + iteration`;
- no ensemble-member perturbation of primary RNG is introduced;
- every per-seed final Advantage gate passes with weighted NRMSE <= `0.75`;
- every per-seed final AveragePolicy gate passes with weighted mean TV <= `0.12`;
- cross-seed mean TV <= `0.15`;
- cross-seed p95 TV <= `0.35`;
- the generated evidence reports `per_seed_fit_pass = true`, `cross_seed_pass = true`, and `r7_3_pass = true`;
- no strategic acceptance threshold is relaxed.

Passing this bridge means only `R7_3_STRATEGY_QUALITY_640 = PASS (PROVISIONAL)` and authorizes R7.4 engineering. It does **not** mean exact R7.3 certification passed.

## Practical risk interpretation

The exact-reproduction failure does not imply a catastrophic action such as folding AA preflop. The measured divergence is in learned-policy/training metrics. Large-margin poker decisions should normally be more robust than near-indifference mixed decisions, but aggregate TV metrics cannot prove that any particular hand/state is unchanged. Borderline or mixed-frequency decisions are the most plausible place for action probabilities to move.

For that reason, action-level sentinel/regression checks for strategically extreme and canonical states are required before final table release. They are defense-in-depth and do not replace the held-out strategic gates.

## Release invariant

Until exact reproducibility debt is closed:

- `R7_3_EXACT_REPRODUCIBILITY = OPEN/DEFERRED`;
- `READY FOR TABLES = false`;
- no artifact or report may label R7.3 as fully certified solely because the provisional 640 strategy bridge passed.
