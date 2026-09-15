# SpinCore Current Work

Date: 2026-09-15
Status: **FUNCTIONAL PATH VALIDATED ON RYZEN — FIRST SUBSTANTIVE TRAINING READY**

## Goal

Make the multi-year DeepSpin project actually work as SpinCore. Preserve mature legacy knowledge; replace only components with a concrete correctness, learning-quality, or compute-efficiency reason.

## First functional SpinCore — decisions closed

- **Scenario distribution:** restored legacy real SpinGo model: 3-handed + true HU, nine blind levels from 10/20 to 100/200, separate empirical blind weights, blind-conditioned stack distributions, 1500 total chips, random dealer/live/dead seats.
- **Payout scope:** one winner-take-all policy family first, reused initially across payout variants. No separate 70/30, 50/30/20 training now.
- **Utility:** chip EV remains the strategic objective. Legacy state-dependent `chip_delta / current_bb` target scaling is replaced by one global positive scale `chip_delta / 1500`. State inputs such as stack/BB and pot/BB remain BB-relative because that geometry is strategically meaningful.
- **Representation:** compact SPNNIV1 exact-state-derived neural boundary. Do not restore the full legacy 292-float vector.
- **Action scope:** preserve the mature DeepSpin seven labels with their actual historical context semantics. Preflop uses 2BB open, 2.5BB+ isolation, 5BB+ 3-bet families and limp/multi-raise restrictions; postflop uses 33/50/75/100% pot-after-call plus legacy near-all-in collapse.
- **Deep CFR traversal:** standard external sampling (`exact_opponent_levels=0`).
- **Average-policy memory:** ordinary sampled game trajectories after advantage fitting, matching the mature DeepSpin mechanism and avoiding R7.5 exact-expansion blow-up.
- **Regret fallback:** fitted advantage networks with all legal outputs <= 0 use masked softmax over raw advantages; uniform is reserved for genuinely untrained initialization.
- **Domains:** separate `THREE_HANDED` and `TRUE_HEADS_UP` brains with realistic per-domain sampling.

## Functional path

Core files now include the restored scenario sampler, lean utility/representation/action scopes, legacy-faithful C++ action resolver, Deep-CFR trainer, resumable checkpoints, offline inference agent and self-play runner.

The trainer now also supports:

- sparse periodic checkpoints via `--checkpoint-every`;
- extending an existing finalized checkpoint with `--resume --additional-iterations N` so useful training is not thrown away merely because an initial budget finished.

## Ryzen evidence — 2026-09-15

The user's Ryzen pilot completed successfully from commit `e3f5d318c0bdc97c207742dc4d5a81f380b87357` using Python 3.12.3, Torch 2.13.0+cpu, NumPy 2.3.5 and two Torch threads.

Five iterations / 1000 roots completed in **28.8 s trainer wall time**, versus about 63.6 s on the 2-thread GitHub runner. Peak RSS was about **389 MB**. The process used about **193% CPU**, confirming that the current path is effectively using about two cores and still leaves substantial Ryzen parallel capacity available if we later need it.

The run produced 545 3H roots + 455 HU roots, 68,860 3H nodes + 42,323 HU nodes, 12,598 3H advantage samples + 8,906 HU, and 1,081 3H strategy samples + 390 HU. The real blind distribution was exercised, including late HU levels through 80/160.

The finalized pilot checkpoint also passed 100-hand offline self-play: 68 3H + 32 HU hands, 325 decisions, no illegal action and successful end-to-end inference. This particular tiny trained policy ended almost every hand preflop/flop (310 preflop, 15 flop, no turn/river), largely because 1000 roots is far too little and its learned policy still shoves/folds excessively. This is not accepted as strategy quality evidence; the earlier GitHub 1000-root seed realization did reach all streets, showing the simulator itself is not structurally preventing later streets.

## Why the first substantive profile is 600 roots/iteration

This is deliberately legacy-first rather than arbitrary. With the restored HU prevalence, 600 roots/iteration split to about 327 3H and 273 HU roots. Because each root traverses every live player, that yields about **1527 advantage traversals per iteration**. The sampled-policy rule yields about **255 policy episodes per iteration**.

The mature DeepSpin defaults were 3 x 512 = **1536 advantage traversals** and **256 policy episodes** per iteration. Therefore 600 current roots/iteration almost exactly preserves the old proven per-iteration sampling scale while using the repaired modern state/action/training path.

## First substantive training profile

`tools/run_lean_functional_first_training.sh` runs:

- 200 iterations;
- 600 roots/iteration = **120,000 roots** total;
- external sampling (`exact_opponent_levels=0`);
- 100,000-sample reservoir capacity per memory;
- 50 advantage optimizer steps/iteration, batch 256;
- 400 final average-policy steps, batch 256;
- checkpoint every 5 iterations;
- 5000-hand offline self-play after training.

At measured Ryzen throughput, this should be an order-of-one-hour job, not a days/weeks experiment. The checkpoint can be extended later without restarting training from zero.

## Historical failure lesson

The earlier DeepSpin trained for roughly three months and still made gross mistakes. Do not answer bad play with 'train longer' before checking game/evaluator semantics, state representation, sampling, regret behavior, action mapping and inference parity. Known historical defects included board-only made-hand interpretation, a uniform all-nonpositive regret fallback and architecture/runtime drift.

## Do not do

- Do not resume Dense-reference i3-i5 merely to complete an old matrix.
- Do not use fixed-10/20 PF0-PF4 evidence as the final global selector.
- Do not restart the old R8/gate chain.
- Do not create payout-specific trainings now.
- Do not reopen representation/action tournaments without concrete play evidence.

## Immediate next milestone

Run the 120k-root first substantive training on the Ryzen. After it finishes, evaluate whether the policy has actually moved away from the 1000-root shove/fold-heavy behavior and then perform one direct strategic comparison before deciding whether to extend this same checkpoint. Do not open another architecture audit unless the resulting play exposes a concrete defect.
