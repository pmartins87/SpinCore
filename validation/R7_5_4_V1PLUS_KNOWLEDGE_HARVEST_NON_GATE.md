# R7.5.4 — V1+ Knowledge Harvest (NON-GATE)

Status: **READ-ONLY / NON-GATE / NO STRATEGY ADMISSION**

Purpose: preserve the scientifically valid information learned during the V1+ architecture reset while honoring the Phase2C2 causal failure and the frozen selection of `C0_V1_FROZEN_CONTROL / SPNNIV1` for R7.5.4.

This document does **not** reopen V1+, create Phase2C3, alter the active candidate, or change any R7.5.4 acceptance criterion.

## 1. Frozen active baseline

Until R7.5.4 completes and R7.5.5 explicitly freezes a production representation/action protocol, the active strategic candidate remains:

- architecture: `C0_V1_FROZEN_CONTROL`;
- representation: `SPNNIV1`;
- historical V1 action protocol and frozen sizing candidates;
- original R7.5.4 seeds, thresholds, budgets, policies and referee contract.

The following are **not admitted** into the active candidate:

- `SPNNIV3`;
- H2/H3 representation changes;
- the Phase2C2 range-reach target kernel or its weighting;
- changes to K, seeds, thresholds or acceptance gates;
- Phase2C3 or another hidden continuation of the V1+ reset.

## 2. What Phase2C0 and Phase2C1 proved

### Phase2C0 — exact structural factorization

For the tested preflop states, the reach probability of the observed public history factorizes, up to numerical precision, into:

`known Hero reach × opponent-A hand reach × opponent-B hand reach × card-removal constraint`.

The frozen C0 evidence completed 16/16 required cases, with maximum factorization error `5.55e-17` against the `1e-12` gate and zero filler/board error.

The important practical observation is that the true joint posterior remains broad: the effective joint support was approximately 2.37M–4.70M assignments (about 3.50M mean). Therefore a tiny sampled representation such as K64 can remain noisy even when the underlying posterior is well behaved.

The same reach component can instead be represented deterministically by two 2,450-entry vectors (39,200 bytes total in float64).

### Phase2C1 — exact incremental propagation

C1 established that those full-hand reach vectors can be propagated action by action: after an observed public action, only the acting opponent's vector is multiplied by `P(observed action | hand, public state)`.

The incremental state is mathematically equivalent to the integral factorization validated by C0, while retaining exact public-state invariants.

These are valid structural results even though C2 later failed.

## 3. What Phase2C2 rejected — and what it did not reject

Phase2C2 was mechanically valid but failed its causal/end-to-end gate. The range-reach arm produced worse COMMON grouped mean TV than control (`0.250565` vs `0.243976`, about 2.70% worse), with bootstrap 95% CI for the difference `[-0.013964, 0.000698]`.

Official classification:

`STRUCTURAL_RANGE_REACH_CAUSAL_EFFECT_NOT_SUPPORTED_SELECT_V1_FALLBACK`

Therefore the claim rejected by C2 is:

> **Injecting exact range-reach into the target through the tested C2 kernel materially stabilizes the end-to-end learner.**

C2 does **not** establish that exact hidden-range inference is wrong, harmful, or useless. C0/C1 already proved the underlying range-reach machinery correct. The failed component is the tested coupling of that information into the learning target.

This distinction is mandatory in all future interpretation of the V1+ experiments.

## 4. V1+ knowledge that may be used now without changing V1 behavior

C0/C1 may be retained as **oracle and instrumentation only** for R7.5.4 and later diagnosis:

1. verify canonical public-history reconstruction from the historical V1 protocol;
2. verify actor identity at every decision node;
3. verify active-player masks;
4. verify the exact legal-action set;
5. verify card-removal, terminality and impossible-state invariants;
6. annotate strategic validation cells with deterministic reach diagnostics such as effective support, concentration/entropy proxies and public-history depth;
7. stratify R7.5.4 TV errors by those diagnostics to locate where residual error concentrates;
8. preserve an exact regression oracle for future architectures.

All such use is observational. It may explain a failure or identify a suspicious state, but it may not change the candidate's action, target, weighting, representation, sizing, threshold or pass/fail result.

## 5. R7.5.4 audit questions enabled by this harvest

Once the historical 36-cell evidence set is complete, the audit should be able to answer, separately:

1. Does the public action trace map exactly to the canonical V1 history?
2. Is the acting player exact at every node?
3. Is the active mask exact?
4. Is the legal-action set exact?
5. Are stack/pot/terminal semantics internally consistent?
6. Do large strategic TV errors cluster by history depth or deterministic range-reach support/concentration?
7. If protocol invariants are exact but TV remains poor, is the residual better attributed to action/sizing representation or solver/training approximation rather than hidden-range reconstruction?

This separation is valuable because it prevents another architecture reset from mixing three different hypotheses: state/history correctness, hidden-range inference, and target/action approximation.

## 6. Decision discipline

The current dense 3H recovery must remain untouched. First recover only the three missing `PF_DENSE_REFERENCE × THREE_HANDED` cells and validate the frozen 36/36 evidence set under the original protocol.

Any annotations or oracle outputs produced from C0/C1 are supplementary evidence only. Existing R7.5.4 strategic gates remain unchanged, including the frozen COMMON/NATIVE TV criteria and the frozen sizing/action comparison protocol.

Only R7.5.5 may freeze a production representation/action protocol.

## 7. Practical conclusion

The correct fallback is **V1 behavior with V1+ knowledge preserved around it**, not a return to an intellectually "raw" V1.

V1+ gave the project three durable pieces of information:

- the observed-history hidden-range reach can be represented and propagated exactly without K-sampling noise;
- the posterior support is genuinely huge, explaining why small sampled hidden-state approximations can be unstable;
- exact inference alone did not make the tested learning target stable, so inference correctness and target usefulness must be evaluated separately.

That knowledge should now improve diagnosis, validation and future architecture design while the active R7.5.4 candidate remains strictly V1-frozen.

## Provenance

- Phase2C0 artifact SHA-256: `55e83be4fd8776e0fcdb63e7d4400ed05aff8c48213898ad8f1abe3713a35876`.
- Phase2C2 artifact SHA-256: `e4b60b0c2826af1751f75eb1d6efe4fd2d86bccf47fc95fa0ac2ffa9b0d04299`.
- V1+ closure recorded on main: `eb48df109fdecbf54bd2d2f0ad8b37a6d94577a1`.
- Current R7.5.4A dense-3H recovery baseline: `30aab24558750fee1f5da0821ecd6fe8a8c8db2d`.
- Governance source: `validation/GOVERNANCE_FRONTIER_REFRESH_20260822_V1PLUS_ARCH_RESET.json`.
