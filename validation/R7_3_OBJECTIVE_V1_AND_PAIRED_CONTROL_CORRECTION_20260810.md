# R7.3 objective V1 and paired-control correction — 2026-08-10

`READY FOR TABLES = NO`.

## 1. Advantage-objective E2E V1 is invalid as CFR evidence

The physical file `validation/R7_3_ADVANTAGE_OBJECTIVE_E2E_256.json` was produced successfully, but inspection of the runner found a state-transition bug in the diagnostic itself.

`DeepCFRDomainSession.reset_advantage_network()` correctly marks the neural behavior policy as not ready because the freshly initialized network is not yet a valid regret estimator. The authoritative training method `DeepCFRDomainSession.train_advantage()` sets `behavior.ready = True` and `counters["advantage_ready"] = 1` after training.

V1 of `tools/run_r7_3_advantage_objective_e2e.py` trained with a custom `_train_custom_step()` loop and bypassed that authoritative transition. Consequently, after every reset the next CFR collection still used the exact zero-regret uniform bootstrap instead of the trained AdvantageNet.

The very small and nearly identical baseline/candidate cross-seed TVs in V1 are therefore not evidence that the auxiliary objective solved R7.3. They are an artifact of the diagnostic remaining on uniform behavior.

V2 fixes this by explicitly restoring the authoritative post-fit readiness state, counting custom Advantage optimizer steps, recording `behavior_ready_after_fit` at every checkpoint, and failing closed if the invariant is violated. The corrected physical workflow is `31427105766`.

## 2. Shared-deck partial-exact + Advantage ensemble is diagnostic, not acceptance-paired

The earlier `R7_3_PARTIAL_EXACT_ENSEMBLE_SIZE{1,2,4}_256.json` experiment intentionally used a common deck stream and separated traversal RNGs. Its result remains useful for causal diagnosis, but it is not directly paired to the generation-2 acceptance contract.

The controlled results were:

| Advantage ensemble | mean TV | p95 TV |
|---:|---:|---:|
| 1 | 0.134403 | 0.565635 |
| 2 | 0.150080 | 0.592874 |
| 4 | 0.154886 | 0.582120 |

The mean can approach the frozen `0.15` threshold under this controlled design, while the p95 remains far above the frozen `0.35` threshold. Increasing the Advantage ensemble size did not improve the result.

A new paired runner now preserves the generation-2 deck formula and the recovered single live RNG stream for traversal, the primary Advantage member, and final AveragePolicy training. Extra ensemble members are side fits on frozen memory and do not advance the primary RNG state. Its workflow is `31427702314`.

## 3. Compatibility regression caught by smoke gate

The first paired-ensemble and final-policy-ensemble workflow attempts failed in smoke before any physical candidate was run. The cause was a source compatibility regression: `run_r7_3_replicated_640_candidate.py` still imports `HISTORICAL_PARAMS_PER_NETWORK`, while the current `run_r7_3_diagnostic.py` no longer exported that symbol.

No numerical evidence was produced by those failed attempts; fallback JSON files are failure markers only. The compatibility constant was restored as `152_434`, matching the historical recorded parameter count. Both workflows were relaunched and their corrected smoke gates must pass before physical evidence is accepted.

## 4. Current acceptance-scale reference for partial exact

The current authoritative physical level-2 640 evidence is `validation/R7_3_PARTIAL_EXACT_LEVEL2_640_CANDIDATE.json`:

- mean TV: `0.4361102283`
- p95 TV: `0.8882443905`
- both per-seed Advantage and AveragePolicy fit gates: PASS
- frozen cross-seed gates: FAIL

The strong-fit variant with up to 8192 Advantage optimizer steps/iteration reached:

- mean TV: `0.4128931165`
- p95 TV: `0.8717077374`
- per-seed fit gates: PASS
- frozen cross-seed gates: FAIL

Thus additional optimizer capacity helps modestly but is not sufficient. This supersedes any stale narrative number for the 640 level-2 candidate.

## 5. Current decision rule

No experimental mechanism is promoted because of a diagnostic-only shared-deck result. The next promotion decision must use paired evidence under the authoritative deck/RNG contract, and any production semantic change still requires explicit versioning plus checkpoint/resume recertification.
