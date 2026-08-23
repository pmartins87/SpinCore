# R7.5 Architecture Reset — V1+ Phase2B8 Lagged Preflop Anchor Training Screen

Status: **FROZEN BEFORE PHASE2B8 OUTPUTS**  
Date: 2026-08-23

## 1. Why this experiment exists

Phase2B6 established that damping behavior only on preflop continuation states has a real end-to-end causal effect on final AveragePolicy stability. Phase2B7 then localized the remaining residual: under COMMON_LEARNER, preflop continuation contributes `0.5493029004171274` of residual TV mass and `0.4742268041237113` of all `TV > 0.35` states, so the frozen route is `PRECOMMIT_EARLY_PREFLOP_LAGGED_TARGET_OR_ANCHOR_SCREEN`.

Phase2B7 also revealed a critical side effect. The Phase2B6 uniform floor was never applied at root, yet the final root AveragePolicy mean TV worsened from `0.2226316130920923` to `0.25663072380695223` and root p95 became `0.6509606611106077`. Root now contributes `0.29383299506889526` of residual TV mass and `0.422680412371134` of the tail. NATIVE_LEARNER corroborates the same pattern.

Therefore Phase2B8 must not increase the uniform floor. It tests whether **temporal anchoring to the previous learned behavior** can retain the beneficial continuation stabilization while avoiding additional strategic flattening toward uniform and avoiding further root damage.

## 2. Frozen causal candidate

Candidate name: `LAGGED_BEHAVIOR_ANCHOR_025`.

At every behavior-policy query:

- `PREFLOP_ROOT`: use the current native H2 four-member uncertainty-damped behavior, unchanged.
- `FLOP/TURN/RIVER`: use the current native behavior, unchanged.
- `PREFLOP_CONTINUATION` after at least one non-forced preflop event: return

`0.75 * current_native_policy + 0.25 * lagged_native_policy`.

The lagged policy is the behavior ensemble that was active **before the most recent Advantage refit**.

Iteration semantics are frozen:

1. before any Advantage model exists, both current and lagged behavior are uniform by the canonical wrapper;
2. after iteration 1 refit, current = iteration-1 ensemble and lagged = initial uniform behavior;
3. after iteration 2 refit, current = iteration-2 ensemble and lagged = iteration-1 ensemble;
4. after iteration 3 refit, current = iteration-3 ensemble and lagged = iteration-2 ensemble, but no further training traversal occurs.

Thus iterations 1 and 2 are behaviorally equivalent to the Phase2B6 25% uniform-floor control. The first intended causal divergence from Phase2B6 occurs in iteration 3, where the 25% anchor becomes the previous learned policy instead of uniform.

No anchor is applied during heldout inference. Final AveragePolicy is evaluated directly.

## 3. Frozen inputs and compute

- Representation: `H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL`.
- Domain: `THREE_HANDED`.
- Action candidate: `PF0_CONTROL_33_75_AI`.
- Training seeds: `1342191342`, `1801739323`.
- Heldout seeds: `2029384436`, `1150634112`.
- First `1024` frozen heldout states per evaluation seed.
- Iterations: `3`.
- Chance coverage: `4 x 64 = 256` roots/iteration.
- Total roots/seed: `768`.
- Exact opponent levels: `2`.
- Advantage reservoir: `100000`.
- Strategy reservoir: `100000`.
- Advantage/model fit budgets, model architecture, optimizer, batch size, LR and action abstraction: unchanged from Phase2B6/Phase2A.
- Exact Phase2B6 control result SHA-256: `33ec6ba89823dae632b7af935def17444379c96a28e59478c0b7c91f1ec3659a`.
- Exact Phase2B7 localization result SHA-256: `ff55a5a047d62952e505b8e4d59d79d4016f30b6696a339318bc696dd6f77fe6`.

Phase2B6 is the exact control and must **not** be retrained.

## 4. Equivalence-before-divergence validity check

Because the candidate is mathematically identical to Phase2B6 through the end of iteration 2, Phase2B8 must compare its iteration-1 and iteration-2 telemetry with the exact local Phase2B6 seed results.

For both seeds and iterations 1-2, the following must match exactly or to numerical tolerance `1e-12` where floating arithmetic is involved:

- roots added;
- Advantage samples added;
- Strategy samples added;
- behavior calls subject to the continuation intervention;
- ensemble weighted NRMSE.

Any mismatch invalidates the experiment before the iteration-3 causal readout is accepted.

This is a stronger check than merely verifying final hashes: it proves the new code does not introduce an unintended intervention before the frozen divergence point.

## 5. Primary and corroborative readouts

Primary: `COMMON_LEARNER` final AveragePolicy cross-seed TV on the two frozen heldouts.

Corroborative: `NATIVE_LEARNER`.

For each learner/heldout report mean and p95 TV. Also report pooled equal-heldout mean TV, paired bootstrap baseline-minus-candidate CI, and the Phase2B7 regions using the same authoritative SPNNIV3 parser:

- `PREFLOP_ROOT`;
- `PREFLOP_CONTINUATION_1`;
- `PREFLOP_CONTINUATION_2PLUS`;
- `FLOP`;
- `TURN`;
- `RIVER`.

Frozen Phase2B6 COMMON reference values:

- pooled mean TV: `0.18934816676149685`;
- root mean TV: `0.25663072380695223`;
- combined preflop-continuation mean TV: `0.1778058850139139`;
- heldout means: `0.18810851478911766`, `0.19058781873387604`;
- heldout p95s: `0.48880605139812816`, `0.5286989995226848`.

Historical hard gates remain unchanged: mean TV `<= 0.15` and p95 TV `<= 0.35` on both required heldouts.

## 6. Precommitted screen decision

`LAGGED_ANCHOR_EFFECT_SUPPORTED` requires **all** of:

1. all local Advantage NRMSE gates pass;
2. all COMMON/NATIVE AveragePolicy fitting gates pass;
3. COMMON mean TV improves on both heldout seeds versus exact Phase2B6;
4. COMMON pooled mean TV improves by at least `0.015` absolute **or** `8%` relative;
5. paired equal-heldout 95% bootstrap CI for Phase2B6-minus-Phase2B8 COMMON TV is strictly positive;
6. neither COMMON heldout p95 worsens by more than `0.02`;
7. NATIVE pooled mean TV does not worsen by more than `0.01`;
8. COMMON root mean TV does not worsen by more than `0.005` versus `0.25663072380695223`;
9. COMMON combined preflop-continuation mean TV does not worsen by more than `0.005` versus `0.1778058850139139`;
10. the equivalence-before-divergence check passes for both seeds and iterations 1-2.

If all above pass and both historical hard heldout gates also pass, classify `LAGGED_ANCHOR_STABILITY_ELIGIBLE_PENDING_STRENGTH`. This still does **not** select an architecture: a separate precommitted strategic-strength comparison against the certified stable V1 control remains mandatory.

If the causal screen passes but hard stability still fails, classify `LAGGED_ANCHOR_EFFECT_SUPPORTED_BUT_STILL_UNSTABLE` and route to `LOCALIZE_RESIDUAL_AFTER_LAGGED_ANCHOR`.

If the causal screen fails, classify `LAGGED_ANCHOR_EFFECT_NOT_SUPPORTED` and route to `REASSESS_ROOT_ANCHOR_OR_DIRECT_TARGET_VARIANCE_REDUCTION`. No post-hoc lag weight tuning is authorized from this failure.

## 7. Prohibitions

- no 50%, 75%, or 100% uniform-floor escalation;
- no alternate lag weight after seeing Phase2B8 outputs;
- no seed shopping;
- no threshold relaxation;
- no dropped heldout/domain;
- no retraining the Phase2B6 control;
- no heldout regeneration;
- no production training;
- no architecture winner selection from this screen alone;
- `READY FOR TABLES = NO`.

## 8. Strategic-strength firewall

Stability and poker strength remain separate gates. A lagged anchor can only become a representation/solver candidate after independently satisfying the hard stability gates. Only then may it enter a precommitted strategic-strength comparison against the certified stable V1 control. A stability improvement that comes from strategically weak or overly inert behavior is not acceptable.
