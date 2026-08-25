# R7.5 architecture reset — Phase2C2 closure and V1 fallback selection

Date: 2026-08-25

Status: **V1PLUS_ARCHITECTURE_RESET_CLOSED — CERTIFIED_STABLE_V1_FALLBACK_SELECTED**

Production training authorized: **NO**

Ready for simulator tables: **NO**

## Binding evidence

The final fair-control Phase2C2 result is:

```text
file:   R7_5_ARCH_RESET_V1PLUS_PHASE2C2_RANGE_REACH_TARGET_KERNEL_CAUSAL_PILOT.json
sha256: e4b60b0c2826af1751f75eb1d6efe4fd2d86bccf47fc95fa0ac2ffa9b0d04299
execution SHA: c7b37b0e4876e9d6eb01183974502da3c3519969
status: STRUCTURAL_RANGE_REACH_CAUSAL_EFFECT_NOT_SUPPORTED_SELECT_V1_FALLBACK
```

The execution was scientifically valid. All eight Advantage gates and all eight final-policy fit gates passed. The frozen control used one uniformly selected member of the same randomized stratified K64 cell set, rotated to the first cell, while the candidate used the arithmetic mean of the same K64 set. Root targets, continuation traversal budgets, seeds, roots, reservoirs, optimizers and heldouts were paired as precommitted.

## Causal adjudication

The primary COMMON learner did not improve:

```text
control pooled mean TV:       0.2439756399
candidate pooled mean TV:     0.2505651700
control-minus-candidate:     -0.0065895301
relative improvement:        -2.7009%
paired bootstrap 95% CI:     [-0.0139636856, 0.0006979155]
```

Only evaluation seed `2029384436` improved, by `0.00048356`; evaluation seed `1150634112` worsened by `0.01366262`. The candidate p95 increased in both COMMON heldouts. Continuation depth 2+ improved in only one heldout. Neither COMMON heldout passed the unchanged hard stability gates.

The NATIVE diagnostic improved by only `0.00142599` mean TV (`0.6225%`), with paired 95% CI `[-0.00568140, 0.00862159]`. This interval crosses zero and cannot override the failed primary COMMON gates.

Therefore:

```text
causal effect supported:          NO
full x4 confirmation authorized: NO
Phase2C3 authorized:             NO
V1+ architecture winner:         NONE
```

## Frozen closure consequence

The precommitted causal-fail branch is now binding:

```text
SELECT_CERTIFIED_STABLE_V1_FALLBACK_AND_CLOSE_V1PLUS_ARCHITECTURE_RESET
```

The successor investigation ends here. Phase2C0 and Phase2C1 remain valid structural findings, but Phase2C2 shows that the bounded range-reach target kernel did not translate those identities into a material end-to-end stability improvement. No x4 confirmation, Phase2C3, larger K, post-hoc stratification, seed replacement, threshold change or estimator-repair reopening is permitted.

The selected fallback is the historical `C0_V1_FROZEN_CONTROL` / `SPNNIV1` lineage. Its certified R7.4 three-handed 640-root stability evidence remains:

```text
cross-seed mean TV: 0.0899957567  PASS
cross-seed p95 TV:  0.2001979053  PASS
```

This selection is a provisional architecture fallback for completing R7.5. It is not an R7.5.5 production freeze and does not infer strategic strength from stability.

The known V1 representation debts remain explicit: absolute physical card identities are embedded directly, public history is the coarse last-32 `(street, action_type)` stream, actor/sizing-rich full history is absent, and the neural action width is six rather than the SPNNIV3 width of ten. The fallback decision accepts none of those properties as strategically optimal. It records that every tested richer successor failed the mandatory stability/admission route, while V1 retains certified stability.

## Next finite gate

R7.5.4 is now eligible for a fail-closed V1 binding audit followed by the already-frozen action/sizing strategic revalidation. The existing action stack was originally bound to `C0_V1_FROZEN_CONTROL`, but its exact source, checkpoint and evaluator identities must be re-audited before a new physical execution or reuse claim.

Only R7.5.4 may select the action abstraction. Only R7.5.5 may adjudicate the remaining V1 representation debt and freeze the production representation/action pair. R8.3–R8.5 remain blocked, and `READY FOR TABLES` remains `NO`.
