# SpinCore Current Work

Date: 2026-09-15
Status: **LEAN FUNCTIONAL PATH IMPLEMENTED — RUN CHEAP SMOKE BEFORE SCALING**

## Goal

Make the multi-year DeepSpin project actually work as SpinCore. Preserve mature legacy knowledge; replace only components with a concrete correctness, learning-quality, or compute-efficiency reason.

## First functional SpinCore — decisions now closed

- **Scenario distribution:** restore the legacy real SpinGo model: 3-handed + true HU, nine blind levels from 10/20 to 100/200, separate empirical blind weights, blind-conditioned stack distributions, 1500 total chips, random dealer/live/dead seats.
- **Payout scope:** train one winner-take-all policy family first and reuse it initially across payout variants. No separate 70/30, 50/30/20 training now.
- **Utility:** chip EV remains the strategic objective. Legacy state-dependent `chip_delta / current_bb` target scaling is replaced on the functional path by one global positive scale `chip_delta / 1500`. This preserves chip-EV action ordering while avoiding blind-dependent target magnitudes in the shared neural approximator. Input features such as stack/BB and pot/BB remain normalized by BB because those describe strategically scale-relative state; that is a separate issue from utility scaling.
- **Representation:** compact SPNNIV1 exact-state-derived neural boundary. Do not restore the full legacy 292-float vector. Preserve its semantic definitions as diagnostic/reference knowledge and add a specific semantic helper only if a concrete weakness justifies it.
- **Action scope:** start from the mature legacy seven-action vocabulary rather than invent another abstraction: `FOLD`, `CHECK_CALL`, `POT_33`, `POT_50`, `POT_75`, `POT_100`, `ALL_IN`. The current ten-slot universal resolver already maps these to exact legal actions, clamps to min/max raise, deduplicates aliases, and maps near-stack raises to all-in. Same seven-action baseline is allowed preflop and postflop initially; prune/expand later only for evidence-backed strategic or compute gain.
- **Deep CFR traversal:** use standard external sampling for the first functional path (`exact_opponent_levels=0` by default). This preserves exact expansion at the traverser's choices while sampling opponent actions, avoiding the enormous branching cost of the later partial-exact certification experiments.
- **Regret fallback:** fitted advantage networks with all legal outputs <= 0 use stable masked softmax over raw advantages, preserving ranking. Uniform behavior is reserved for the genuinely untrained initial state.
- **Domains:** separate `THREE_HANDED` and `TRUE_HEADS_UP` brains remain, but roots are budgeted according to the observed legacy HU/3H prevalence and each brain samples its own empirical blind/stack distribution.

## Implemented functional path

- `python/spincore/legacy_scenario_data.json`
- `python/spincore/legacy_scenario.py`
- `python/spincore/lean_training_scope.py`
- `python/spincore/lean_representation_scope.py`
- `python/spincore/lean_action_scope.py`
- `python/spincore/lean_action_policy.py`
- `python/spincore/lean_functional_training.py`
- `tools/run_lean_functional_training.py`

The functional trainer is resumable after each iteration and intentionally omits the old four-member uncertainty ensemble, referee matrix, bootstrap gates, fixed-10/20 scenario cycle and homologation-style machinery.

## Historical failure lesson

The earlier DeepSpin trained for roughly three months on the Ryzen and still made gross mistakes. Do not answer bad play with 'train longer' before checking game/evaluator semantics, state representation, sampling, regret behavior, action mapping, and training/runtime parity. Known historical defects included board-only made-hand interpretation and a uniform all-nonpositive regret fallback.

## What remains before a long Ryzen run

Only one architecture-affecting item remains on the critical path: **training/runtime parity**. Legacy `user_deepspin.cpp` is useful integration material but cannot be reused unchanged because it expects the old 292-feature/7-output contract. The new OpenHoldem runtime must produce the same SPNNIV1 semantics and seven-active-slot universal action mapping used during training.

This runtime work does **not** block a cheap local training smoke, because the smoke's purpose is only to prove the newly consolidated training mechanics execute on the real scenario distribution and to measure actual roots/second. It does block calling a long-trained model table-ready.

Crusher/hardcoded and solver-v2/184 remain reference assets, not independent audit gates.

## CI note

A broad historical regression previously had 384 Python tests pass and one failure caused only by a stale frozen Git-blob hash for an intentionally evolved R7.5 representation file. C++ regression passed. Do not turn that historical certification hash into a new workstream.

## Do not do

- Do not resume Dense-reference i3-i5 just to complete an old matrix.
- Do not use fixed-10/20 PF0-PF4 evidence as the final global selector.
- Do not restart the old R8/gate chain.
- Do not launch another representation/action tournament before the functional baseline exists.
- Do not create payout-specific trainings now.

## Immediate next milestone

Run the two-root lean smoke (one effective root budget for each domain), inspect only whether mechanics are correct and how long a root actually takes, then choose the first meaningful Ryzen training budget from measured throughput. No weeks-long run is authorized before that measurement.
