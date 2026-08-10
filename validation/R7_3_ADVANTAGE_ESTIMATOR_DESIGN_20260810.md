# R7.3 Advantage estimator design record — 2026-08-10

`READY FOR TABLES = NO`. Frozen R7.3 gates remain unchanged.

This record narrows the remaining R7.3 failure to the Advantage external-sampling estimator and defines the next estimator-design ladder without changing production semantics prematurely.

## 1. Why Advantage sampling is now the primary target

The causal chain is supported by multiple independent physical experiments:

1. changing CFR memories explains much more cross-seed policy disagreement than changing AveragePolicy optimizer/init;
2. common hidden cards barely reduce that disagreement;
3. exact uniform-policy controls prove that own-reach sampling fragments support and that opponent-action external sampling injects Advantage target noise;
4. stronger neural fitting alone improves the 640 acceptance metric only modestly;
5. the controlled four-mode downstream screen shows that four Advantage trajectories reduce fitted cross-seed mean TV far more than four strategy trajectories, and also improve p95 materially.

The downstream path screen (`31366433008`, evidence `a9c57fe6e3c9149ed3010ead280912295bd4f5f6`) produced:

- baseline mean/p95: `0.305382 / 0.870543`
- strategy_x4: `0.275642 / 0.865904`
- advantage_x4: `0.219118 / 0.690974`
- both_x4: `0.197598 / 0.726534`

All individual fit gates passed and the baseline-vs-strategy_x4 Advantage checkpoint isolation delta was exactly zero. Persisted diagnosis: `ADVANTAGE_EXTERNAL_SAMPLING_VARIANCE_MATERIAL`.

## 2. Exact opponent-expectation Advantage oracle

Workflow `31368837895`, evidence commit `45c68d2028ac658ae12870c97b9bf758e47f2a89`, schema `SPINCORE_R7_3_EXACT_ADVANTAGE_FEASIBILITY_V1`.

The oracle keeps traverser actions enumerated exactly as the recovered external-sampling collector does, but also enumerates opponent actions. Returned utility is the exact sigma-weighted opponent expectation. Downstream Advantage training-distribution mass is weighted by explicit opponent reach, making this the mathematical expectation of sampling one opponent action from sigma.

Under the exact zero-regret uniform policy, four unique HU deals and both traversers required:

- `1,265,152` nodes;
- `188,440` Advantage samples before aggregation;
- exact opponent-expectation phase `15.91 s` on the GitHub CPU runner;
- max depth `45`;
- all eight target-root traversals completed below the one-million-node cap.

This establishes an exact oracle for small-scale estimator evaluation.

## 3. Independent sampled paths measured against the exact oracle

On the same four deals, sampled external-sampling memories were compared directly to exact target vectors and exact regret-matching policies.

| sampled Advantage trajectories | sampled unique | exact weight coverage | target relative RMSE | regret-matching mean TV | p50 TV | p95 TV | weighted greedy agreement |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 73 | 0.042562 | 0.835858 | 0.381860 | 0.062393 | 1.000000 | 0.480239 |
| 4 | 540 | 0.110336 | 0.676023 | 0.270441 | 0.020846 | 1.000000 | 0.733842 |
| 8 | 1,185 | 0.157548 | 0.686965 | 0.257149 | 0.012940 | 1.000000 | 0.747338 |

Every sampled observation belonged to the exact support (`sampled_precision_against_exact = 1.0`).

The most important engineering result is the dose response:

- 1 -> 4 paths cuts exact-oracle regret-matching mean TV by roughly **29%** and improves weighted greedy-action agreement from `0.48` to `0.73`;
- 4 -> 8 yields only a further small mean-TV reduction (`0.2704 -> 0.2571`), while p95 remains saturated at `1.0`;
- four paths therefore capture a large fraction of the readily available independent-replication benefit, consistent with the downstream four-mode screen where `advantage_x4` was the strongest isolated lever.

This independently supports the choice of x4 rather than blindly escalating to x8 or more at acceptance scale.

## 4. Exact enumeration is an oracle, not yet the production solution

Full exact opponent expectation is much more expensive than four sampled trajectories. Extrapolating the bounded bootstrap tree directly to every root and every CFR iteration would dramatically increase node count and raw Advantage sample volume. The exact oracle is therefore used to evaluate cheaper unbiased estimators rather than being silently promoted.

The desired production estimator should approach exact target/regret-matching quality with a bounded compute and memory factor.

## 5. Partial-exact opponent estimator now under physical test

Workflow `31369138285` tests a versioned estimator candidate at 256 unique roots/seed with exact-opponent levels `0`, `1`, and `2`.

Semantics:

- level `0`: recovered external sampling; one opponent action is sampled at every non-traverser node;
- level `1`: enumerate the next opponent decision exactly on each path, then resume external sampling;
- level `2`: enumerate the next two opponent decisions exactly on each path, then resume sampling.

At enumerated opponent branches, downstream Advantage samples receive an explicit probability-mass multiplier. Later sampled opponent decisions remain represented by Monte-Carlo occurrence probability. This preserves the expected training distribution rather than overrepresenting low-probability enumerated branches.

The level-0 implementation is required to reproduce the previously persisted controlled baseline (`mean TV 0.3053824902`, `p95 0.8705430627`) to `1e-9`; otherwise the workflow fails. That turns the previous physical result into a regression oracle for the new estimator implementation.

Each level then trains strong Advantage and AveragePolicy networks and reports fitted cross-seed TV plus total node cost. This measures **variance reduction per unit tree work**, not merely whether more computation helps.

## 6. Acceptance-scale candidates running in parallel

While the estimator-design screen runs, the already-supported x4 schedule is being tested at full 640 roots so no wall-clock time is wasted.

Workflow `31368447316` runs simultaneously:

- `advantage_x4`: four Advantage paths, one strategy path, separated diagnostic RNG streams;
- `both_x4`: four Advantage paths, four strategy paths, same separated streams.

Workflow `31368894934` runs a third 640 candidate:

- `advantage_x4` under the **recovered coupled RNG contract**.

The coupled run is important because it answers whether the x4 benefit survives without introducing a new RNG-stream schema. If it performs comparably to separated x4, it becomes operationally preferable because only the sampling schedule—not the RNG state structure—needs versioning and checkpoint/resume recertification.

All three candidates keep the frozen R7.3 gates unchanged, use the acceptance independent hidden-deal schedule, use strong Advantage fitting, and enlarge the reservoir to `400000` so x4 data is not silently discarded by the old 100k cap.

## 7. Promotion hierarchy

The next production-facing decision is deliberately finite:

1. Prefer **coupled advantage_x4** if its 640 result matches or beats separated x4 closely enough, because it preserves the recovered RNG-state contract.
2. Otherwise prefer **separated advantage_x4** if it has the best frozen-gate balance, especially p95.
3. Use **both_x4** only if its 640 improvement justifies the added strategy cost and it does not sacrifice the p95 gate.
4. If independent x4 remains far from the gates, use the partial-exact screen to choose bounded opponent enumeration.
5. If partial exact is still insufficient, move to a versioned stratified/antithetic or paired/common-random-number opponent estimator and benchmark it against the exact oracle above.
6. Do not resume brute-force unique-root scaling until the per-deal path estimator is stable enough that additional deals add coverage rather than mostly new Monte-Carlo path noise.
