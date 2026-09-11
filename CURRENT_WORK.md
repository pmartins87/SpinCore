# SpinCore Current Work

Date: 2026-09-11
Status: **CORRECTIVE LEGACY-FIRST IMPLEMENTATION — NO HEAVY TRAINING YET**

## Why heavy training is paused

R7.5.4 action-abstraction training/evaluation was discovered to be restricted to SB=10 / BB=20. That is useful only as localized evidence and cannot represent the full SpinGo tournament distribution or select the final global policy/action abstraction.

The project already had a multi-year legacy archive, `Tentativas anteriores de SpinGo.zip`, containing a substantially richer SpinGo-specific foundation. That archive is the baseline; SpinCore is an evolution of it.

## Decisions now fixed for the first functional SpinCore

- Restore the real legacy 3H/HU scenario distribution: nine blind levels, separate empirical blind weights, blind-conditioned stack distributions, 1500 total chips, live/dead-seat and dealer randomization.
- Train **one winner-take-all chip-EV policy family** first.
- Use the same WTA-trained policy initially across payout variants. Do not multiply first-release training into separate 70/30, 50/30/20, etc. runs.
- Keep ICM support in the solver for possible later multi-place specialization, but do not spend that compute before the WTA agent is strong and functional.
- Use the compact exact-state-derived **V1 neural representation** for first release rather than restoring the legacy 292-float flat vector. Preserve the legacy 292 feature definitions as poker-semantic diagnostic/reference knowledge and restore only a specific derived feature if a concrete weakness justifies it.
- Change legacy behavior only for a concrete correctness/quality reason.
- Do not treat the old `/BB` payoff normalization as a bug yet; its cross-blind learning effect remains a focused audit question.

## Corrections already implemented

1. `python/spincore/legacy_scenario_data.json` preserves the legacy empirical blind/stack tables.
2. `python/spincore/legacy_scenario.py` adapts those tables to current `Episode` objects and restores the historical sampling semantics for both `THREE_HANDED` and `TRUE_HEADS_UP`.
3. `python/spincore/deep_cfr.py` no longer regresses to uniform play when a fitted advantage net predicts all legal advantages <= 0. It now preserves the repaired legacy behavior: stable masked softmax over raw legal advantages. The genuinely untrained zero-regret initial state remains uniform separately.
4. `python/spincore/lean_training_scope.py` encodes `CHIP_EV_WTA_V1` and `WTA_SHARED_ACROSS_PAYOUTS_V1`; multi-place specialization is disabled for the first release.
5. `docs/LEGACY_292_FEATURE_AUDIT.md` closes the first-release representation question: do not reintroduce the full 292-float input; keep compact V1 and preserve legacy feature semantics as reference knowledge.

## Why the 292-feature vector is not being restored wholesale

The legacy vector contains useful poker knowledge, but it is highly redundant: 104 raw-card one-hots plus 81 hand/draw/board semantic dimensions plus 65 action-context/history dimensions. More importantly, the old networks were extremely wide: about 2.14M parameters for AdvantageNet and 1.09M for PolicyNet. Shrinking only the 292 input would reduce that total only modestly; the large hidden layers and traversal volume were the main compute burden.

Current V1 instead keeps the exact canonical game state underneath and presents the model with structured card tokens, numeric/categorical state and public-history tokens. The recovered V1 network is about 152k parameters per model. This substantially reduces model cost and the semantic bug surface without throwing away the exact game state.

## Historical failure lesson

The earlier DeepSpin trained continuously for roughly three months on the Ryzen and still made gross mistakes. That cannot be treated as ordinary undertraining. Known structural risks include evaluator/feature semantic errors, the historical uniform-regret fallback, excessive complexity, and trainer/runtime drift. Therefore additional training is not the remedy until these architecture-affecting failure modes are checked.

## Immediate remaining audit — finite, not academic

Only questions capable of changing the first functional implementation remain on the critical path:

- `/BB` target normalization: keep or remove based on its effect on shared cross-blind learning;
- action set: reconcile legacy 7 actions with current SpinCore action representation under the restored full tournament distribution;
- training/runtime parity: ensure the model receives the same semantics during training and inference;
- Crusher/hardcoded and solver-v2 assets: consult only where they can materially improve one of the three questions above, not as independent audit projects.

## CI note

The latest broad `main_regression` run had 384 Python tests pass and one failure caused by an old frozen Git-blob hash for `r7_5_representation_v3_final_policy.py`. C++ regression passed. That failure is a stale historical-freeze/certification contract, not evidence that the restored sampler, WTA scope or V1 decision is broken. Do not spend a new validation campaign on it; demote/reconcile the stale freeze when cleaning the critical CI path.

## Do not do now

- Do not resume dense 3H i3-i5 merely to complete an old matrix.
- Do not run the PF0-PF4 10/20-only comparison as a final selector.
- Do not start old R8 heavy training.
- Do not create separate payout-specific trainings.
- Do not launch another broad representation tournament.
- Do not add certification/reproducibility gates unless they can materially change playing quality or catch a real correctness bug.

## Next milestone

Close `/BB`, action-set reconciliation and training/runtime parity, then create the **first lean functional training runner** using the restored real scenario distribution, compact V1 representation and WTA chip-EV scope. Start with one cheap sanity run; if mechanics and learning are sane, scale directly on Ryzen instead of opening another chain of academic gates.
