# R7.3 exact path-variance controls — 2026-08-10

`READY FOR TABLES = NO`. Frozen R7.3 gates are unchanged.

This record separates path-estimator facts from acceptance-candidate controls. A post-run audit found that the replicated-candidate V1 640 runs used a different deterministic hidden-deal schedule than the authoritative reference; their metrics remain valid experiments but are not paired deck-identical deltas. See `validation/R7_3_REPLICATED_V1_DECK_CONTROL_CORRECTION_20260810.md`.

## 1. Own-reach support-density control

Workflow `31366894171`, evidence `3f5130561f0e3d83e65f33b451af86ce80dfa04d`.

Same 256 deals, exact zero-regret uniform behavior, no neural fitting:

| own-reach paths | poker-isomorphic Jaccard | mean LCFR-weight coverage | shared-target TV |
|---:|---:|---:|---:|
| 1 | `0.033946` | `0.083230` | `0` |
| 2 | `0.041047` | `0.113135` | `0` |
| 4 | `0.052395` | `0.149171` | `~3.7e-19` |
| 8 | `0.074383` | `0.204593` | `~2.8e-18` |

1 -> 8 raises Jaccard `2.191x` and LCFR-weight coverage `2.458x` while the true targets stay identical. Own-reach Monte Carlo itself fragments observed AveragePolicy support.

## 2. Advantage target-variance control

Workflow `31366996254`, evidence `d1ab3ebc7a905ce2b164e1bf1dee1d5c3efd0a87`.

Same exact uniform policy and hidden deals; only opponent actions are sampled:

| Advantage paths | weight coverage | target relative RMSE | regret-match mean TV | p95 |
|---:|---:|---:|---:|---:|
| 1 | `0.077203` | `1.009432` | `0.421004` | `1.0` |
| 2 | `0.089112` | `0.965445` | `0.407431` | `1.0` |
| 4 | `0.114344` | `0.912991` | `0.382970` | `1.0` |
| 8 | `0.158538` | `0.881045` | `0.371266` | `1.0` |

External-sampling Advantage target variance is real. Independent replication helps the mean but not the extreme tail.

## 3. Exact expectation controls

### Exact own reach

Workflow `31367407567`, evidence `a16e043fffda04f2c2fa228611e3e352d7ca39b8`:

- four deals / both target players: `1,265,152` nodes;
- `188,440` target-state samples;
- `116,192` unique raw observations;
- eight sampled paths cover only `2.107%` of exact action-path support.

### Exact opponent-expectation Advantage

Workflow `31368837895`, evidence `45c68d2028ac658ae12870c97b9bf758e47f2a89`:

- four deals / both traversers: `1,265,152` nodes;
- `188,440` Advantage samples;
- exact phase `15.91 s`.

| sampled paths | exact weight coverage | target relative RMSE | regret-match mean TV | greedy agreement | p95 |
|---:|---:|---:|---:|---:|---:|
| 1 | `0.042562` | `0.835858` | `0.381860` | `0.480239` | `1.0` |
| 4 | `0.110336` | `0.676023` | `0.270441` | `0.733842` | `1.0` |
| 8 | `0.157548` | `0.686965` | `0.257149` | `0.747338` | `1.0` |

The 1 -> 4 gain is large; 4 -> 8 is small. Independent x8/x16 is therefore not promoted without new evidence.

## 4. Two-iteration downstream path decomposition

Workflow `31366433008`, evidence `a9c57fe6e3c9149ed3010ead280912295bd4f5f6`.

All individual fit gates passed. The baseline-versus-`strategy_x4` Advantage checkpoint NRMSE delta was exactly zero, proving clean isolation.

| mode | mean TV | p95 TV |
|---|---:|---:|
| baseline | `0.305382` | `0.870543` |
| strategy_x4 | `0.275642` | `0.865904` |
| advantage_x4 | `0.219118` | `0.690974` |
| both_x4 | `0.197598` | `0.726534` |

This identifies Advantage path replication as the strongest isolated **short-horizon** path lever.

## 5. Replicated-candidate V1 640 results

The V1 640 outputs all failed cross-seed gates while passing individual fits:

| V1 candidate | mean TV | p95 TV | evidence |
|---|---:|---:|---|
| separated Advantage x4 | `0.459596` | `0.898250` | `94b5e423fa51e1dad8445e6ce36b8832d8161648` |
| separated both x4 | `0.458853` | `0.908883` | `871967f777f7cec17479ed3ec9f476543452912d` |
| coupled Advantage x4 | `0.451112` | `0.893292` | `87547311076fd6a015b7d855de1a9c26124b924f` |

However, V1 used:

```text
(seed << 32) ^ (iteration << 16) ^ root_index_within_iteration
```

rather than the authoritative:

```text
seed*1_000_003 + global_root*97 + iteration
```

with continuous global root. Therefore the V1 values show that x4 was not remotely sufficient on those physical samples, but they are **not** a paired measure of improvement versus corrected 640.

## 6. Deck-exact paired x4 V2

Workflow `31414208511` is now running the corrected coupled Advantage x4 candidate with the exact authoritative deck schedule, 640 roots/seed, four Advantage paths, one strategy path, recovered coupled RNG, strong fitting and 400k reservoir.

V2 is the acceptance-scale result to use when quantifying x4's paired effect.

## 7. Active variance-reduction redesign

- Partial-exact V2 `31412806987`: authoritative level 0 versus exact opponent levels 1/2.
- Full-exact upper bound `31412933368`: bounded effectively complete opponent expectation.
- Common-random-number screen `31413103901`: independent 1/4 versus common 1/4, with byte-identical iteration-1 memory invariant.
- Five-iteration compounding `31413646505`: baseline and x4 regret-policy divergence after every refit.
- Antithetic x4 `31413970227`: ordinary independent x4 versus four marginally correct, quarter-turn correlated Uniform streams.

## 8. Decision rule

- Use V2, not V1, for paired x4 acceptance conclusions.
- Prefer common-path or antithetic correlation if it reduces iterative variance at equal/lower cost.
- Prefer bounded partial enumeration if it captures most of the full-exact upper-bound gain.
- If full exact is weak, pivot from opponent sampling to regret-sign sensitivity, policy-support discontinuity, target aggregation and control variates.
- Keep brute-force roots and independent x8/x16 paused.
