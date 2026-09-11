# SpinCore Current Work

Date: 2026-09-10
Status: **CORRECTIVE LEGACY-FIRST IMPLEMENTATION — NO HEAVY TRAINING YET**

## Why heavy training is paused

R7.5.4 action-abstraction training/evaluation was discovered to be restricted to SB=10 / BB=20. That is useful only as localized evidence and cannot represent the full SpinGo tournament distribution or select the final global policy/action abstraction.

The project already had a multi-year legacy archive, `Tentativas anteriores de SpinGo.zip`, containing a substantially richer SpinGo-specific foundation. That archive is the baseline; SpinCore is an evolution of it.

## Decisions now fixed for the first functional SpinCore

- Restore the real legacy 3H/HU scenario distribution: nine blind levels, separate empirical blind weights, blind-conditioned stack distributions, 1500 total chips, live/dead-seat and dealer randomization.
- Train **one winner-take-all chip-EV policy family** first.
- Use the same WTA-trained policy initially across payout variants. Do not multiply first-release training into separate 70/30, 50/30/20, etc. runs.
- Keep ICM support in the solver for possible later multi-place specialization, but do not spend that compute before the WTA agent is strong and functional.
- Change legacy behavior only for a concrete correctness/quality reason.
- Do not treat the old `/BB` payoff normalization as a bug yet; its cross-blind learning effect remains a focused audit question.

## Corrections already implemented

1. `python/spincore/legacy_scenario_data.json` preserves the legacy empirical blind/stack tables.
2. `python/spincore/legacy_scenario.py` adapts those tables to current `Episode` objects and restores the historical sampling semantics for both `THREE_HANDED` and `TRUE_HEADS_UP`.
3. `python/spincore/deep_cfr.py` no longer regresses to uniform play when a fitted advantage net predicts all legal advantages <= 0. It now preserves the repaired legacy behavior: stable masked softmax over raw legal advantages. The genuinely untrained zero-regret initial state remains uniform separately.
4. `python/spincore/lean_training_scope.py` encodes `CHIP_EV_WTA_V1` and `WTA_SHARED_ACROSS_PAYOUTS_V1`; multi-place specialization is disabled for the first release.
5. Focused regression guards were added for the restored scenario sampler, shared WTA policy family, and nonpositive-regret fallback. No broad new certification campaign is required.

## Historical failure lesson

The earlier DeepSpin trained continuously for roughly three months on the Ryzen and still made gross mistakes. That cannot be treated as ordinary undertraining. Known structural risks include evaluator/feature semantic errors, the historical uniform-regret fallback, excessive complexity, and trainer/runtime drift. Therefore additional training is not the remedy until these architecture-affecting failure modes are checked.

## Immediate remaining audit — finite, not academic

Only questions capable of changing the implementation remain on the critical path:

- 292-feature legacy observation: which features are sound, which were repaired, and whether current SpinCore lost strategically useful information;
- `/BB` target normalization: keep or remove based on its effect on shared cross-blind learning;
- action set: reconcile legacy 7 actions with current SpinCore action representation under the restored full tournament distribution;
- training/runtime parity: ensure the model receives the same semantics during training and inference;
- Crusher/hardcoded and solver-v2 assets: preserve only strategically useful knowledge, not their complexity for its own sake.

## Do not do now

- Do not resume dense 3H i3-i5 merely to complete an old matrix.
- Do not run the PF0-PF4 10/20-only comparison as a final selector.
- Do not start old R8 heavy training.
- Do not create separate payout-specific trainings.
- Do not add certification/reproducibility gates unless they can materially change playing quality or catch a real correctness bug.

## Next milestone

Finish the finite architecture-affecting audit above, then create the **first lean functional training runner** using the restored real scenario distribution and WTA chip-EV scope. Start with a cheap sanity run; if mechanics and learning are sane, scale directly on Ryzen instead of opening another chain of academic gates.
