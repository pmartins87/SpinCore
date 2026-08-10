# R7.3 same-memory Advantage fit instability — 2026-08-10

`READY FOR TABLES = NO`. Frozen gates are unchanged.

## 1. Same frozen memory, independently fitted AdvantageNets

Workflow `31414959700`, evidence `d0fa315ee79af013a7e7e3294b0877a0e656f820`.

One fixed iteration-1 external-sampling reservoir was collected under exact zero-regret uniform behavior:

- 256 roots;
- `10,565` Advantage samples / seen;
- `61,966` traversal nodes.

Four AdvantageNets were trained from scratch on **exactly the same frozen reservoir**. Only initialization and minibatch RNG differed:

| replica | final weighted NRMSE | steps |
|---|---:|---:|
| 0 | `0.483531` | 2560 |
| 1 | `0.483770` | 2304 |
| 2 | `0.472173` | 2816 |
| 3 | `0.487427` | 2048 |

Every replica is comfortably inside the frozen `0.75` gate and around historical-quality `~0.5` fit.

Yet across all six replica pairs on the same 2,048 in-memory observations:

- hard-regret-matching mean TV averaged **`0.224349`**;
- p95 averaged **`0.757529`**;
- identical positive-regret action support occurred on only **`55.49%`** of observations.

Diagnosis: `ADVANTAGE_FIT_REGRET_SIGN_INSTABILITY_MATERIAL`.

Representative replica0-vs1:

- mean TV `0.225397`;
- p95 `0.739181`;
- max `1.0`;
- legal-action sign disagreement `15.51%`;
- identical positive support `55.76%`.

**Conclusion:** even with the CFR target memory held completely fixed, historical-quality Advantage regression error is large enough in strategically sensitive directions to generate substantially different CFR behavior policies.

## 2. Near-zero regret/sign fragility

For replica0-vs1, observations where at least one fitted model has a legal predicted regret near zero are disproportionately responsible for hard-regret policy disagreement:

- normalized min-|regret| <= `0.05`: `17.38%` of observations, `22.06%` of total TV mass;
- <= `0.10`: `29.49%` of observations, `39.23%` of TV mass;
- <= `0.25`: `54.20%` of observations, `70.05%` of TV mass.

A post-hoc sensitivity probe added a scale-normalized positive floor after prediction, without changing training or production semantics:

- hard rule: mean `0.225397`, p95 `0.739181`;
- epsilon `0.05`: mean `0.203056`, p95 `0.641930`;
- epsilon `0.10`: mean `0.185336`, p95 `0.574292`.

Across replica pairs, epsilon `0.10` reduces mean TV to roughly `0.82–0.83x` hard-regret matching. Hard sign/support thresholding is therefore a material amplifier, but not the complete source: the smoothed policies still disagree strongly.

## 3. Exact 2x2 fit-randomness factorial

Workflow `31415792326`, evidence `555df805e5f14814b1f3e742481bcff110d6cc49`, schema `SPINCORE_R7_3_ADVANTAGE_FIT_FACTORIAL_V1`.

The same frozen Advantage reservoir was trained with a 2x2 design:

- A_X: init A, minibatch stream X
- A_Y: init A, minibatch stream Y
- B_X: init B, minibatch stream X
- B_Y: init B, minibatch stream Y.

Hard-regret policy disagreement:

- **same init, different minibatch order:** mean TV `0.201320`;
- **different init, same minibatch order:** `0.221737`;
- **different init and minibatch order:** `0.224532`;
- init/minibatch component ratio `1.1014`.

Diagnosis: `ADVANTAGE_INIT_AND_MINIBATCH_VARIANCE_MIXED`.

Initialization is slightly larger, but not enough to call it dominant. Stabilizing only initialization or only minibatch order is unlikely to solve the function-approximation variance. A deterministic/common fit contract must control **both** if it is to remove stochastic fitting as a source.

## 4. Five-iteration compounding: baseline versus Advantage x4

Workflow `31413646505` has now completed both physical jobs. Baseline evidence `83e057dee6c2a66d46bf1eb64409855c51631748`; Advantage-x4 evidence `03308e8dfdb359332e081737691941ea96e3133d`.

After each CFR iteration, the freshly fitted AdvantageNets were converted through production hard regret matching on a common current-iteration corpus.

| iteration | baseline mean TV | Advantage x4 mean TV | x4/baseline | x4 reduction |
|---:|---:|---:|---:|---:|
| 1 | `0.476820` | `0.300327` | `0.6299` | `37.0%` |
| 2 | `0.578266` | `0.520984` | `0.9009` | `9.9%` |
| 3 | `0.562001` | `0.496018` | `0.8826` | `11.7%` |
| 4 | `0.553982` | `0.475241` | `0.8579` | `14.2%` |
| 5 | `0.538547` | `0.522414` | `0.9700` | **`3.0%`** |

The p95 is `1.0` for **both modes at every iteration**.

Final AveragePolicy cross-seed metrics:

- baseline: mean `0.453395`, p95 `0.948678`;
- Advantage x4: mean `0.436853`, p95 `0.929471`.

This resolves an important ambiguity. More Advantage paths substantially reduce the **first** fitted policy disagreement, but that advantage is progressively overwhelmed by later independent neural fits / CFR feedback; by iteration 5 the fresh-regret-policy mean TV is only `3%` better than baseline. x4 is therefore reducing sampled-target noise, but it is not stabilizing the learned regret approximator strongly enough across repeated resets.

The first-iteration result is also revealing. Under iteration-1 uniform behavior, shared strategy targets on shared observations are exactly identical, yet fitted regret-policy TV is already `0.4768` baseline and `0.3003` even with x4. This aligns directly with the same-memory experiment: **AdvantageNet regression variance plus hard regret matching is capable of creating large policy divergence before CFR feedback has had time to compound anything.**

## 5. Current engineering consequence

R7.3 has two upstream mechanisms that must be handled separately:

1. **target/memory variance** from sampled CFR trajectories — bounded partial opponent enumeration is the strongest current treatment;
2. **Advantage function-approximation variance** — independent fits to the same targets produce different regret signs/supports.

Already closed/deprioritized as primary treatments:

- common opponent-path random numbers: only modest improvement;
- antithetic x4: essentially flat mean and worse p95;
- exhaustive opponent enumeration: huge support/reservoir burden and worse end-to-end stability;
- independent x8/x16: exact oracle shows diminishing returns.

Active fit-stability experiments:

- common Advantage initialization + common per-iteration minibatch RNG across algorithm seeds (`31415931101`), with traversal and final AveragePolicy randomness still seed-specific;
- eight-replica same-memory Advantage ensemble diagnostic (`31416468310`), comparing disjoint 1-, 2- and 4-model raw-Advantage averages before unchanged hard regret matching.

Active target-variance/acceptance experiments:

- deck-exact coupled x4 V2 `31414208511`;
- partial-exact level1/level2 640 `31415605322`;
- level2 strong-fit 640 `31415642047`.

The next production candidate should not simply stack all interventions. First identify whether deterministic/common fitting or a small ensemble materially suppresses same-memory behavior variance; then combine that proven fit-stability mechanism only with the cheapest bounded partial-exact estimator still required by acceptance-scale evidence.
