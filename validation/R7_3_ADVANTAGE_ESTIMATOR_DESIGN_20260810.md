# R7.3 Advantage estimator design record — 2026-08-10

`READY FOR TABLES = NO`. Frozen R7.3 gates remain unchanged.

## 1. Established causal picture

Physical experiments now support the following hierarchy:

1. changing CFR memories explains much more cross-seed disagreement than changing AveragePolicy init/optimizer;
2. hidden-card stream differences explain little;
3. off-support extrapolation is material but downstream;
4. stronger Advantage fitting helps, but is insufficient at 640;
5. exact bootstrap controls prove that own-reach sampling fragments strategy support and opponent external sampling injects Advantage target noise;
6. at two CFR iterations, four Advantage paths strongly improve fitted cross-seed stability;
7. at five iterations / 640 roots, that x4 improvement largely disappears.

The remaining problem is therefore no longer “does external sampling have variance?” It does. The harder question is **which lower-variance estimator prevents small early errors from compounding through repeated regret matching and neural refits?**

## 2. Short-screen path decomposition

Workflow `31366433008`, evidence `a9c57fe6e3c9149ed3010ead280912295bd4f5f6`:

| mode | mean TV | p95 TV |
|---|---:|---:|
| baseline | `0.305382` | `0.870543` |
| strategy_x4 | `0.275642` | `0.865904` |
| advantage_x4 | `0.219118` | `0.690974` |
| both_x4 | `0.197598` | `0.726534` |

All individual fit gates passed; the baseline-vs-strategy_x4 Advantage checkpoint isolation delta was exactly zero. Advantage replication is the strongest isolated short-screen lever.

## 3. Exact opponent-expectation oracle

Workflow `31368837895`, evidence `45c68d2028ac658ae12870c97b9bf758e47f2a89`.

Four unique HU deals and both traversers:

- exact nodes `1,265,152`;
- exact Advantage samples `188,440`;
- exact phase `15.91 s`;
- max depth `45`.

Sampled external-sampling memories versus exact expectation:

| paths | exact weight coverage | target relative RMSE | regret-match mean TV | p95 | greedy agreement |
|---:|---:|---:|---:|---:|---:|
| 1 | `0.042562` | `0.835858` | `0.381860` | `1.0` | `0.480239` |
| 4 | `0.110336` | `0.676023` | `0.270441` | `1.0` | `0.733842` |
| 8 | `0.157548` | `0.686965` | `0.257149` | `1.0` | `0.747338` |

This gives a direct engineering rule: x4 captures most of the inexpensive independent-replication improvement; x8 adds little average benefit and still fails catastrophically on a tail of states. Therefore independent x8/x16 is not the next preferred design.

## 4. Acceptance-scale x4 results

Three full 640 candidates have now completed:

| candidate | mean TV | p95 TV | individual fits | evidence |
|---|---:|---:|---|---|
| separated Advantage x4 | `0.459596` | `0.898250` | PASS | `94b5e423fa51e1dad8445e6ce36b8832d8161648` |
| separated both x4 | `0.458853` | `0.908883` | PASS | `871967f777f7cec17479ed3ec9f476543452912d` |
| recovered-coupled Advantage x4 | **`0.451112`** | `0.893292` | PASS | `87547311076fd6a015b7d855de1a9c26124b924f` |

For context, corrected 640 is `0.477649 / 0.902403` and strong-Advantage 640 is `0.464474 / 0.886204`.

### Design consequence

Coupled x4 is the best x4 mean and would have been operationally attractive because it preserves the recovered RNG state structure. But it is still far from `0.15 / 0.35`, and its p95 is not better than the strong-Advantage run. `both_x4` adds substantial collection/training cost with no tail benefit.

**Plain independent replication is closed as the primary convergence strategy.** It remains a possible component of a later estimator, but not the solution by itself.

The contrast between the two-iteration x4 screen and five-iteration 640 runs is evidence of **iterative variance amplification**: a reduction that is clearly visible early is largely erased as independently learned regret policies feed back into later collections.

## 5. Partial-exact V1 — rejected diagnostic control

Workflow `31369138285` passed build, CTest and all 26 Python tests, then rejected its own physical experiment because an experimental reimplementation of level 0 failed an exact comparison to a separately persisted baseline. The fallback evidence intentionally states `runner_failed_before_report=true`.

No V1 positive-level result is accepted.

This is a diagnostic-design failure, not a solver regression.

## 6. Partial-exact V2 — active, workflow `31412806987`

V2 fixes the control architecture:

- level 0 is executed through the authoritative recovered `ExternalSamplingCollector` itself;
- level 1 enumerates the next opponent decision exactly, probability-weights downstream Advantage samples, then resumes sampling;
- level 2 enumerates the next two opponent decisions;
- comparisons are against a fresh authoritative baseline in the same process and dependency image;
- the old persisted baseline is informational only.

The smoke passed; the physical 256-root V2 screen is running. This measures variance reduction per additional tree work rather than merely adding repeated trajectories.

## 7. Full-exact upper bound — active, workflow `31412933368`

A 64-root/seed screen compares the authoritative baseline to effectively full opponent-action enumeration (`exact level 128`, beyond observed max depth). This is deliberately an upper bound.

Interpretation:

- large improvement => opponent external-sampling variance remains a sufficient high-leverage target, and the problem becomes finding the cheapest bounded approximation to exact expectation;
- small improvement => external sampling is real but not sufficient; move downstream to regret-sign/policy-support dynamics and neural target processing.

## 8. Common-random-number path estimator — active, workflow `31413103901`

Modes:

- `independent_1`
- `independent_4`
- `common_1`
- `common_4`

Common modes derive opponent-action RNG from `(iteration, root, traverser, replicate)` rather than algorithm seed. With shared decks and iteration-1 uniform behavior, common-mode Advantage memories must be byte-identical between seeds; this is a hard invariant.

The purpose is not to make two final training seeds artificially identical. It is to test a standard variance-reduction idea: correlate the stochastic estimator noise so random early branch choices do not become an avoidable source of divergence that regret matching later amplifies.

If material, a production version would require a counter-based RNG contract and checkpoint/resume versioning.

## 9. Current estimator ladder

The finite design order is now:

1. **Common random numbers** if they substantially reduce iterative divergence at low cost.
2. **Bounded partial opponent enumeration** if it captures most of the full-exact upper bound efficiently.
3. If needed, stratified/antithetic opponent-action sampling or paired control-variate estimators benchmarked against the exact oracle.
4. If the full-exact upper bound itself is weak, pivot away from opponent sampling to iteration-level regret-sign sensitivity, target aggregation and policy-support discontinuity.

Not promoted without new evidence:

- more unique roots;
- x8/x16 independent replication;
- more strategy-path replication;
- card-representation rewrite;
- simply increasing neural optimizer steps.

Any estimator that eventually clears R7.3 must be explicitly versioned and deterministically recertified for continuous versus stop/restore/continue operation before R7.4 begins.
