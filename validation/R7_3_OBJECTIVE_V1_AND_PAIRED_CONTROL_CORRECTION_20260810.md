# R7.3 objective V1 and paired-control correction — 2026-08-10

`READY FOR TABLES = NO`.

## 1. Advantage-objective E2E V1 is invalid as CFR evidence

The physical file `validation/R7_3_ADVANTAGE_OBJECTIVE_E2E_256.json` was produced successfully, but inspection of the runner found a state-transition bug in the diagnostic itself.

`DeepCFRDomainSession.reset_advantage_network()` correctly marks the neural behavior policy as not ready because the freshly initialized network is not yet a valid regret estimator. The authoritative training method `DeepCFRDomainSession.train_advantage()` sets `behavior.ready = True` and `counters["advantage_ready"] = 1` after training.

V1 of `tools/run_r7_3_advantage_objective_e2e.py` trained with a custom `_train_custom_step()` loop and bypassed that authoritative transition. Consequently, after every reset the next CFR collection still used the exact zero-regret uniform bootstrap instead of the trained AdvantageNet.

The very small and nearly identical baseline/candidate cross-seed TVs in V1 are therefore not evidence that the auxiliary objective solved R7.3. They are an artifact of the diagnostic remaining on uniform behavior.

## 2. Corrected Advantage-objective V2

V2 explicitly restores and asserts neural-behavior readiness after every custom fit. Physical workflow `31427105766` completed successfully and every readiness assertion passed.

Controlled 256-root result:

| mode | mean TV | p95 TV | Advantage fit |
|---|---:|---:|---|
| recovered weighted MSE baseline | `0.294431` | `0.802882` | PASS |
| MSE + policy auxiliary `0.10` | `0.235531` | `0.771797` | **FAIL** |

The candidate reduced mean TV to `0.79995x` baseline, but p95 only to `0.96128x`. More importantly, its final Advantage NRMSE failed the frozen `0.75` gate on both seeds (`0.892394` and `0.836072`), after iteration-2 checkpoint NRMSEs had already risen to `0.939820` and `0.827246`.

The physical diagnosis is therefore `BEHAVIOR_AWARE_ADVANTAGE_OBJECTIVE_NOT_MATERIAL_END_TO_END`. This branch is closed as a primary R7.3 solution. The frozen gate is not changed.

## 3. Shared-deck partial-exact + Advantage ensemble is diagnostic, not acceptance-paired

The earlier `R7_3_PARTIAL_EXACT_ENSEMBLE_SIZE{1,2,4}_256.json` experiment intentionally used a common deck stream and separated traversal RNGs. Its result remains useful for causal diagnosis, but it is not directly paired to the generation-2 acceptance contract.

| Advantage ensemble | mean TV | p95 TV |
|---:|---:|---:|
| 1 | `0.134403` | `0.565635` |
| 2 | `0.150080` | `0.592874` |
| 4 | `0.154886` | `0.582120` |

The mean can approach the frozen `0.15` threshold under this controlled design, while the p95 remains far above the frozen `0.35` threshold. Increasing the raw-Advantage ensemble size did not improve the result.

## 4. Authoritative paired 256 baseline and raw-ensemble matrix

A new paired runner preserves the generation-2 deck formula and the recovered single live RNG stream for traversal, the primary Advantage member, and final AveragePolicy training. Extra ensemble members are side fits on frozen memory and do not advance the primary RNG state.

Corrected workflow `31427702314` passed all smoke gates. Physical size 1 completed and establishes the within-run paired baseline:

- mean TV `0.2456560284`
- p95 TV `0.6287055612`
- all per-seed fit gates PASS

Physical size 2 also completed:

- mean TV `0.2153794914` (`0.8768x` size-1 mean)
- p95 TV `0.6321426630` (`1.0055x` size-1 p95)
- fit gates PASS

Thus size 2 lowers the center of the distribution but does not improve the tail. Size 4 is still physically running and must be compared only with this within-run size-1 baseline. The stale source note saying size 1 should reproduce the older partial-exact 256 screen is incorrect: that older screen used shared decks plus split traversal RNGs and is a different controlled experiment.

## 5. Final AveragePolicy ensemble result

Workflow `31427741160`, evidence commit `70be0290fff93e2f38030ce9010fb7b544e8137e`, trained side AveragePolicy models only after the authoritative partial-exact CFR collection was frozen. Side models did not perturb the primary live RNG stream.

| final policy ensemble | mean TV | p95 TV | mean ratio | p95 ratio |
|---:|---:|---:|---:|---:|
| 1 | `0.245656` | `0.628706` | 1.0000 | 1.0000 |
| 2 | `0.226901` | `0.598557` | `0.92365` | `0.95205` |
| 4 | `0.212912` | `0.599173` | `0.86671` | `0.95303` |

All fit gates pass. Own-memory policy fit improves substantially with four models (roughly `0.077–0.079` weighted TV versus `0.096–0.104` for one model), but cross-seed p95 improves only about `4.7%`.

Conclusion: final-policy approximation/extrapolation is real but **not the dominant source of the p95 tail**. Final AveragePolicy ensembling is not promoted as a primary standalone solution. It remains a possible secondary component only if an upstream candidate gets p95 close enough to the frozen `0.35` boundary that a small additional reduction matters.

## 6. Compatibility regression caught by smoke gate

The first paired-ensemble and final-policy-ensemble workflow attempts failed in smoke before any physical candidate was run. The cause was a source compatibility regression: `run_r7_3_replicated_640_candidate.py` still imports `HISTORICAL_PARAMS_PER_NETWORK`, while the current `run_r7_3_diagnostic.py` no longer exported that symbol.

No numerical evidence was produced by those failed attempts; fallback JSON files are failure markers only. The compatibility constant was restored as `152_434`, matching the historical recorded parameter count. Corrected smoke gates now PASS.

## 7. Current acceptance-scale reference for partial exact

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

## 8. Active next tests

- paired raw Advantage ensemble size 4, workflow `31427702314` — physical run active;
- paired partial-exact Advantage **policy-mixture** size 4, workflow `31428299914` — physical run active; same-memory mapping evidence reduced size-4 p95 from `0.472424` for raw averaging to `0.360312` when hard regret matching is applied per member before probability averaging;
- same-memory Advantage common-mode centering, workflow `31428588011` — physical run active;
- authoritative partial-exact strategy-target support/tail forensic, workflow `31429040423` — active; measures remaining shared-target disagreement by street, legal-action count and history length.

## 9. Current decision rule

No experimental mechanism is promoted because of a diagnostic-only result. Promotion requires paired evidence under the authoritative deck/RNG contract. If upstream policy-mixture gets close to the frozen p95 gate, final-policy ensembling can be tested only as a secondary factorial component. If the p95 remains high, the partial-exact support forensic decides whether the next intervention belongs in the CFR target/behavior update rather than in final-policy smoothing. Any production semantic change still requires explicit versioning plus checkpoint/resume recertification.
