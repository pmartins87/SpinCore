# R7.3 exact path-variance controls — 2026-08-10

`READY FOR TABLES = NO`. Frozen R7.3 gates are unchanged.

This record distinguishes four different facts that must not be conflated:

1. own-reach Monte Carlo fragments AveragePolicy support under a fixed policy;
2. opponent external sampling creates Advantage target variance;
3. independent Advantage replication materially reduces variance in a short controlled experiment;
4. that independent-replication gain is largely lost after five CFR iterations at 640-root acceptance scale.

## 1. Exact own-reach support-density control

Workflow `31366894171`, evidence `3f5130561f0e3d83e65f33b451af86ce80dfa04d`.

Same 256 hidden deals, exact zero-regret uniform behavior, no neural fitting:

| own-reach paths | poker-isomorphic Jaccard | mean LCFR-weight coverage | shared-target TV |
|---:|---:|---:|---:|
| 1 | `0.033946` | `0.083230` | `0` |
| 2 | `0.041047` | `0.113135` | `0` |
| 4 | `0.052395` | `0.149171` | `~3.7e-19` |
| 8 | `0.074383` | `0.204593` | `~2.8e-18` |

1 -> 8 raises Jaccard `2.191x` and LCFR-weight coverage `2.458x` while the true shared policy targets remain identical. The collector itself therefore creates support fragmentation.

## 2. Exact Advantage target-variance control

Workflow `31366996254`, evidence `d1ab3ebc7a905ce2b164e1bf1dee1d5c3efd0a87`.

Same exact uniform policy and hidden deals; only opponent actions are sampled:

| Advantage paths | weight coverage | target relative RMSE | regret-match mean TV | p95 |
|---:|---:|---:|---:|---:|
| 1 | `0.077203` | `1.009432` | `0.421004` | `1.0` |
| 2 | `0.089112` | `0.965445` | `0.407431` | `1.0` |
| 4 | `0.114344` | `0.912991` | `0.382970` | `1.0` |
| 8 | `0.158538` | `0.881045` | `0.371266` | `1.0` |

External-sampling target variance is real. More independent paths help the mean but not the extreme tail.

## 3. Exact expectation feasibility controls

### Own reach

Workflow `31367407567`, evidence `a16e043fffda04f2c2fa228611e3e352d7ca39b8`: four deals required `1,265,152` nodes and `188,440` target-state samples; eight sampled paths covered only `2.107%` of exact action-path support.

### Advantage opponent expectation

Workflow `31368837895`, evidence `45c68d2028ac658ae12870c97b9bf758e47f2a89`: four deals required the same `1,265,152` full-tree nodes and `188,440` Advantage samples; exact phase `15.91 s`.

Sampled paths against the exact Advantage oracle:

| paths | exact weight coverage | target relative RMSE | regret-match mean TV | greedy agreement | p95 |
|---:|---:|---:|---:|---:|---:|
| 1 | `0.042562` | `0.835858` | `0.381860` | `0.480239` | `1.0` |
| 4 | `0.110336` | `0.676023` | `0.270441` | `0.733842` | `1.0` |
| 8 | `0.157548` | `0.686965` | `0.257149` | `0.747338` | `1.0` |

The 1 -> 4 gain is large; 4 -> 8 is small. This is why independent x8/x16 was not promoted after x4 failed acceptance scale.

## 4. Downstream two-iteration path decomposition

Workflow `31366433008`, evidence `a9c57fe6e3c9149ed3010ead280912295bd4f5f6`. All individual fit gates passed and baseline versus `strategy_x4` had exactly identical Advantage checkpoint NRMSEs.

| mode | mean TV | p95 TV |
|---|---:|---:|
| baseline | `0.305382` | `0.870543` |
| strategy_x4 | `0.275642` | `0.865904` |
| advantage_x4 | `0.219118` | `0.690974` |
| both_x4 | `0.197598` | `0.726534` |

This correctly identified Advantage external sampling as the strongest **isolated short-screen** path mechanism.

## 5. Acceptance-scale falsification of plain x4 as the solution

The promoted 640 candidates all passed individual fit gates but failed cross-seed gates:

| candidate | mean TV | p95 TV | evidence |
|---|---:|---:|---|
| separated Advantage x4 | `0.459596` | `0.898250` | `94b5e423fa51e1dad8445e6ce36b8832d8161648` |
| separated both x4 | `0.458853` | `0.908883` | `871967f777f7cec17479ed3ec9f476543452912d` |
| recovered-coupled Advantage x4 | **`0.451112`** | `0.893292` | `87547311076fd6a015b7d855de1a9c26124b924f` |

Corrected 640 reference is `0.477649 / 0.902403`; strong-Advantage 640 is `0.464474 / 0.886204`.

The best x4 mean improvement versus corrected baseline is only about `5.6%`, and the p95 tail is effectively unchanged. `both_x4` adds substantial strategy cost and makes p95 worse. Therefore independent path count is **not** the acceptance-scale fix.

The two-iteration versus five-iteration contrast is evidence that stochastic differences are being amplified by repeated regret matching / neural refitting. A variance reducer that merely lowers single-iteration noise can still fail if independently evolving policies subsequently visit and reinforce different regions of the game tree.

## 6. Current path-variance redesign

Three complementary physical experiments are active:

- partial-exact V2 (`31412806987`): authoritative level 0 versus probability-weighted exact opponent levels 1/2;
- full-exact upper bound (`31412933368`): authoritative baseline versus effectively complete opponent expectation at bounded 64-root scale;
- common-random-number screen (`31413103901`): independent 1/4 paths versus common 1/4 paths, with a byte-identical iteration-1 Advantage-memory invariant.

A fourth experiment (`31413646505`) measures freshly fitted regret-policy cross-seed TV after every one of five CFR iterations for baseline and Advantage x4. Its purpose is to locate **when** the short-screen benefit disappears and whether divergence accelerates after a specific refit/iteration.

## 7. Decision rule

- Prefer common-path/counter-based randomness if correlation suppresses iterative amplification at low cost.
- Prefer bounded partial enumeration if it captures most of the full-exact upper-bound gain per node.
- If full exact is weak, stop optimizing opponent sampling and move to regret-sign sensitivity, policy-support discontinuity, target aggregation and control-variate design.
- Keep unique-root scaling and x8/x16 independent replication paused.
