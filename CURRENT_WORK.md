# SpinCore Current Work

Date: 2026-09-15
Status: **FIRST FUNCTIONAL OFFLINE AGENT WORKS — READY FOR RYZEN THROUGHPUT PILOT**

## Goal

Make the multi-year DeepSpin project actually work as SpinCore. Preserve mature legacy knowledge; replace only components with a concrete correctness, learning-quality, or compute-efficiency reason.

## First functional SpinCore — decisions now closed

- **Scenario distribution:** restored legacy real SpinGo model: 3-handed + true HU, nine blind levels from 10/20 to 100/200, separate empirical blind weights, blind-conditioned stack distributions, 1500 total chips, random dealer/live/dead seats.
- **Payout scope:** one winner-take-all policy family first, reused initially across payout variants. No separate 70/30, 50/30/20 training now.
- **Utility:** chip EV remains the strategic objective. Legacy state-dependent `chip_delta / current_bb` target scaling is replaced by one global positive scale `chip_delta / 1500`. This preserves chip-EV action ordering without blind-dependent target weighting. State inputs such as stack/BB and pot/BB remain BB-relative because that geometry is strategically meaningful.
- **Representation:** compact SPNNIV1 exact-state-derived neural boundary. Do not restore the full legacy 292-float vector. Preserve its semantic definitions as diagnostic/reference knowledge and add a specific derived feature only if a concrete weakness justifies it.
- **Action scope:** preserve the mature DeepSpin seven labels: `FOLD`, `CHECK_CALL`, `POT_33`, `POT_50`, `POT_75`, `POT_100`, `ALL_IN`, but with their actual historical context semantics rather than naïvely treating every label as the same pot fraction on every street. Preflop uses the legacy 2BB open, 2.5BB+ isolation, 5BB+ 3-bet families and limp/multi-raise restrictions; postflop uses 33/50/75/100% pot-after-call plus legacy near-all-in collapse. A dedicated lean C++ resolver implements this without modifying historical R7.5 experiments.
- **Deep CFR traversal:** standard external sampling (`exact_opponent_levels=0`) for the functional path: branch all traverser choices, sample opponent choices.
- **Average-policy memory:** restored the mature DeepSpin mechanism: ordinary sampled game trajectories after the advantage fit. The old R7.5 exact-opponent strategy-memory expansion was generating tens of thousands of unnecessary policy samples and severe compute blow-up in 3H; it is not used by the lean candidate.
- **Regret fallback:** fitted advantage networks with all legal outputs <= 0 use stable masked softmax over raw advantages, preserving ranking. Uniform behavior is reserved for genuinely untrained initialization.
- **Domains:** separate `THREE_HANDED` and `TRUE_HEADS_UP` brains remain. Roots are budgeted using the restored HU/3H prevalence and each brain samples its own empirical blind/stack distribution.

## Implemented functional path

- `python/spincore/legacy_scenario_data.json`
- `python/spincore/legacy_scenario.py`
- `python/spincore/lean_training_scope.py`
- `python/spincore/lean_representation_scope.py`
- `python/spincore/lean_action_scope.py`
- `python/spincore/lean_action_policy.py`
- `include/spincore/lean_action_abstraction.hpp`
- `src/lean_action_abstraction.cpp`
- `python/spincore/lean_solver_actions.py`
- `python/spincore/lean_functional_training.py`
- `tools/run_lean_functional_training.py`
- `python/spincore/lean_functional_agent.py`
- `tools/play_lean_functional_selfplay.py`

The trainer is resumable after every iteration. The offline inference agent loads the finalized AveragePolicy checkpoint and uses the same SPNNIV1 input contract and the same lean C++ legal/exact action resolver as training.

## Functional evidence that matters

The cheap two-root smoke passed on Linux/Python 3.11/Torch 2.13 CPU/NumPy 2.3.5.

A five-iteration benchmark then trained 1000 realistic roots total (545 3H + 455 HU) with fitted advantage policies and sampled policy-memory collection. It completed in about 64 seconds of trainer wall time on a 2-thread GitHub CPU runner, with approximately 400 MB maximum resident memory. The strategy reservoir grew to 1101 3H samples and 395 HU samples instead of the old exact-expansion explosion.

The finalized checkpoint then played 100 complete offline self-play hands through the same solver/action contract with **PASS**: 403 decisions, 68 3H hands, 32 HU hands, decisions on preflop/flop/turn/river, no illegal action, no non-zero-sum terminal result, and all seven active legacy-equivalent action slots represented where legal.

This proves mechanics and training/inference semantic parity inside the SpinCore simulator. It does **not** prove strategic strength: 1000 roots is deliberately far too little to call the agent strong.

## Historical failure lesson

The earlier DeepSpin trained for roughly three months on the Ryzen and still made gross mistakes. Do not answer bad play with 'train longer' before checking game/evaluator semantics, state representation, sampling, regret behavior, action mapping, and inference parity. Known historical defects included board-only made-hand interpretation, a uniform all-nonpositive regret fallback, and architecture/runtime drift.

## What remains before serious quality training

The architecture is now coherent enough to measure the user's Ryzen directly. The next run is a **throughput pilot**, not another architecture tournament. Its purpose is to determine how much strategically useful Deep-CFR work the Ryzen can perform per hour with the corrected functional path, then choose the first substantial training budget accordingly.

Parallel root collection from the mature legacy worker design remains a possible later acceleration because extra throughput can directly buy more strategy quality. Do not implement multiprocessing merely for elegance; use the Ryzen pilot to determine whether single-process throughput is already sufficient or whether worker restoration has high value.

External real-money client attachment is not part of this milestone. The current functional target is the offline SpinCore simulator/inference contract.

## CI note

The lean C++ action semantics and Python path compile and run. The broad historical regression still has one known stale frozen-Git-blob hash failure for an intentionally evolved R7.5 representation file; that certification hash is unrelated to poker correctness and remains non-blocking.

## Do not do

- Do not resume Dense-reference i3-i5 merely to complete an old matrix.
- Do not use fixed-10/20 PF0-PF4 evidence as the final global selector.
- Do not restart the old R8/gate chain.
- Do not launch another representation/action tournament before actual strategic evidence identifies a weakness.
- Do not create payout-specific trainings now.
- Do not launch a days/weeks-long Ryzen run before measuring the corrected functional path locally.

## Immediate next milestone

Run the corrected five-iteration / 1000-root benchmark once on the Ryzen, including the 100-hand offline self-play. Use the measured local wall time to size the first substantial training run. If the Ryzen is materially underused, restore root-level parallel workers before spending a long training budget.
