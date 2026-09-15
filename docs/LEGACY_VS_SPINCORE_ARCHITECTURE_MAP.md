# Legacy DeepSpin vs SpinCore — Architecture Map

Status: **FIRST FUNCTIONAL ARCHITECTURE SELECTED**
Started: 2026-09-09
Updated: 2026-09-15
Purpose: preserve multi-year legacy knowledge and change only what has a concrete reason to improve correctness, strategic quality or compute efficiency.

## 1. Scenario distribution — RESTORED FROM LEGACY

Legacy `deepspin/scenario.py` models the real tournament-state distribution:

- 3-handed and true HU separately;
- blinds 10/20 through 100/200;
- empirical blind weights by mode;
- blind-conditioned empirical stack distributions;
- 1500 total chips;
- random live/dead seats and dealer.

This is restored in `python/spincore/legacy_scenario_data.json` and `python/spincore/legacy_scenario.py`. Fixed-10/20 R7.5.4 work is localized evidence only.

## 2. Action abstraction — LEGACY SEVEN-ACTION BASELINE

Legacy DeepSpin used:

`fold`, `check/call`, 33% pot, 50% pot, 75% pot, 100% pot, all-in.

Decision: use exactly that mature seven-action vocabulary for the first functional SpinCore. `python/spincore/lean_action_scope.py` activates the equivalent seven slots inside the current ten-slot universal resolver. The authoritative C++ resolver clamps fractional targets to the legal min/max raise range, converts stack-capped targets to all-in and deduplicates aliases, so the neural tree does not pay twice for two labels that resolve to the same exact action.

Do not reopen a broad action-abstraction tournament before a functional baseline exists. Prune or expand only when measured strategic gain or branching-cost reduction justifies it under the full real scenario distribution.

## 3. Observation/state — KEEP LEGACY KNOWLEDGE, NOT THE 292-FLOAT INPUT

The archived DeepSpin observation was 292 floats and included useful poker semantics, but also large redundancy and a wide surface for semantic bugs. Its old networks were about 3.23M parameters combined.

Decision: first functional SpinCore uses compact SPNNIV1: structured card tokens, numeric/categorical exact-state-derived fields, legal mask and public-history tokens. The canonical state underneath remains exact. The recovered V1 model is about 152k parameters per network.

Legacy hand-strength/draw/board features remain a semantic checklist. Reintroduce only a specific derived feature when a concrete strategic weakness demonstrates that it is worth the complexity. Detailed rationale: `docs/LEGACY_292_FEATURE_AUDIT.md`.

## 4. Learning algorithm — KEEP DEEP CFR, REMOVE CERTIFICATION OVERHEAD

Useful legacy/current foundation retained:

- external-sampling Deep CFR;
- advantage + average-policy networks;
- reservoir memories;
- exact cloneable C++ game state;
- resumable checkpoints;
- separate 3H/HU domains.

For the first functional path, opponent actions are sampled (`exact_opponent_levels=0`) while traverser actions are expanded exactly. This returns to the practical external-sampling foundation and avoids the huge branching cost introduced by later partial-exact certification experiments.

A historical defect is permanently repaired on the functional path: when a **fitted** advantage network predicts every legal advantage <= 0, use stable masked softmax over raw advantages rather than arbitrary uniform play. Truly untrained initialization remains uniform. The universal-action implementation is `python/spincore/lean_action_policy.py`.

## 5. Utility/objective — WTA CHIP EV, CONSTANT GLOBAL SCALE

Chip EV remains the first-release strategic objective. Multi-place payout specialization is deferred; one WTA-trained policy family is initially reused for all payout variants.

Legacy DeepSpin used `chip_delta / current_bb`. That does not alter action ordering inside one state, but it changes the relative target magnitude supplied to a shared neural approximator across blind levels. With the restored full blind ladder, there is no strategic reason to make one chip worth 10x less to the learning target simply because the current BB is 200 instead of 20.

Decision: `python/spincore/lean_training_scope.py` uses `chip_delta / 1500`, one global positive constant because the tournament has 1500 total chips. This is mathematically the same chip-EV objective up to a constant scale and therefore cannot change exact action ordering, while keeping targets numerically bounded without blind-dependent reweighting.

Important distinction: state inputs such as `stack / BB` and `pot / BB` remain normalized by BB because they describe strategically meaningful relative stack/pot geometry. Only the **utility target** stops using a state-dependent BB divisor.

## 6. Domain allocation — SEPARATE BRAINS, REALISTIC PREVALENCE

The first functional path retains separate `THREE_HANDED` and `TRUE_HEADS_UP` brains. Training root budget is split according to the legacy empirical HU prevalence (~45.48%), while each domain independently samples its real conditional blind/stack distribution.

This avoids forcing one network to reconcile materially different 3H and HU strategy while still spending compute roughly where real games spend hands.

## 7. Runtime/integration — PRESERVE LEGACY INTEGRATION KNOWLEDGE, CHANGE THE CONTRACT

Legacy `user_deepspin.cpp` is substantial OpenHoldem integration work and remains a design/reference asset. It cannot be dropped in unchanged because it expects the old 292-feature/7-output neural contract.

The first functional runtime must reconstruct the exact same SPNNIV1 semantics used during training and map the seven active universal slots to the same exact action resolver. Training/runtime semantic identity is the only remaining architecture-affecting blocker before table use.

## 8. Why historical DeepSpin failed

Known constraints from the multi-year work:

- roughly three months of Ryzen training still produced gross errors, so ordinary undertraining is not an adequate explanation;
- earlier hand/evaluator semantics sometimes confused board-created made hands with Hero-contributed strength;
- an earlier all-nonpositive regret fallback injected uniform, strategically absurd actions;
- representation/runtime complexity and version drift made semantic parity difficult;
- later experimental work expanded compute/certification complexity without first preserving the real tournament-state sampler.

The response is not 'train longer'. It is: correct semantics + real scenarios + leaner representation + repaired regret behavior + practical external sampling + runtime parity.

## 9. Implemented first functional training path

`python/spincore/lean_functional_training.py` and `tools/run_lean_functional_training.py` now combine:

**real legacy SpinGo scenarios + compact V1 + legacy seven actions + external-sampling Deep CFR + repaired regret fallback + WTA chip EV /1500 + separate 3H/HU brains + resumable iteration checkpoints.**

The old four-member uncertainty ensemble, fixed-10/20 matrix, referee completion, bootstrap certification and homologation gates are deliberately not part of this first functional path.

## 10. Next step

Run a two-root local smoke only to prove mechanics and measure true throughput. If it works, choose a meaningful training budget from measured seconds/root and scale on the Ryzen. Do not open another architecture tournament unless actual play or learning evidence identifies a concrete weakness.
