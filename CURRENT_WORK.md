# SpinCore Current Work

Date: 2026-09-15
Status: **FIRST SUBSTANTIVE TRAINING COMPLETED — CHECKPOINT PRESERVED; RYZEN HARDWARE TUNING NEXT; STRATEGY STRENGTH NOT YET ESTABLISHED**

## Goal

Make the multi-year DeepSpin project actually work as SpinCore. Preserve mature legacy knowledge; replace only components with a concrete correctness, learning-quality, or compute-efficiency reason.

## First functional SpinCore — decisions closed

- **Scenario distribution:** restored legacy real SpinGo model: 3-handed + true HU, nine blind levels from 10/20 to 100/200, separate empirical blind weights, blind-conditioned stack distributions, 1500 total chips, random dealer/live/dead seats.
- **Payout scope:** one winner-take-all policy family first, reused initially across payout variants. No separate 70/30, 50/30/20 training now.
- **Utility:** chip EV remains the strategic objective. Legacy state-dependent `chip_delta / current_bb` target scaling is replaced by one global positive scale `chip_delta / 1500`. State inputs such as stack/BB and pot/BB remain BB-relative because that geometry is strategically meaningful.
- **Representation:** compact SPNNIV1 exact-state-derived neural boundary. Do not restore the full legacy 292-float vector.
- **Action scope:** preserve the mature DeepSpin seven labels with their actual historical context semantics. Preflop uses 2BB open, 2.5BB+ isolation, 5BB+ 3-bet families and limp/multi-raise restrictions; postflop uses 33/50/75/100% pot-after-call plus legacy near-all-in collapse.
- **Deep CFR traversal:** standard external sampling (`exact_opponent_levels=0`).
- **Average-policy memory:** ordinary sampled game trajectories after advantage fitting, matching mature DeepSpin and avoiding R7.5 exact-expansion blow-up.
- **Regret fallback:** fitted advantage networks with all legal outputs <= 0 use masked softmax over raw advantages; uniform is reserved for genuinely untrained initialization.
- **Domains:** separate `THREE_HANDED` and `TRUE_HEADS_UP` brains with realistic per-domain sampling.

## First substantive serial baseline — COMPLETED 2026-09-15

The pre-optimization run that had already started was allowed to finish rather than discarding nearly completed work.

Training contract:

- commit used to launch: `85ef4b970e88b78346d0cf306320681bff2cdb94`;
- 200 iterations;
- 600 roots/iteration = **120,000 roots**;
- 65,400 3H roots + 54,600 HU roots;
- 50 advantage optimizer steps/iteration, batch 256;
- 400 final AveragePolicy steps;
- reservoir capacity 100,000;
- external sampling, `exact_opponent_levels=0`;
- WTA chip EV `/1500`;
- checkpoint every 5 iterations.

Final aggregate learning workload:

- **20,838,215 traversal nodes**;
- **3,806,986 advantage samples seen** (2,022,140 3H + 1,784,846 HU);
- **213,935 AveragePolicy samples seen** (156,299 3H + 57,636 HU);
- final policy losses: 1.071089 (3H), 1.179489 (HU).

The empirical sampler was genuinely active across the blind ladder. Rare HU late levels including 100/200 occurred in training; this was not a fixed-10/20 run.

Local durable artifacts:

- run dir: `/home/rz9/spincore_lean_functional/runs/lean_first_training/20260915_131133`;
- checkpoint: `.../checkpoint.pt`;
- report: `.../report.json`;
- self-play: `.../selfplay.json`.

Execution baseline on the Ryzen:

- external elapsed time: **1:25:39**;
- process CPU: **198%**, i.e. roughly two logical CPUs on the 32-thread machine;
- peak RSS: about **1.04 GiB**;
- swaps: 0;
- exit status: 0.

This serial throughput is now a baseline only. It must not be used as the production Ryzen execution profile.

## 5,000-hand offline self-play after training — PASS

The finalized checkpoint completed 5,000 sampled hands with:

- 21,131 decisions;
- 2,770 3H / 2,230 HU hands;
- no illegal selected action;
- zero-sum terminal chip accounting on every hand;
- maximum 17 decisions in one hand;
- street decisions: **15,333 preflop / 3,485 flop / 1,546 turn / 767 river**;
- action counts: `[2928, 8630, 0, 1753, 0, 1053, 0, 828, 272, 5667]`.

Compared with the tiny 1,000-root pilot, the trained policy is materially less degenerate behaviorally: the pilot had 310/325 decisions preflop, no turn/river and 115/325 all-ins; the 120k checkpoint reaches every street and all-in frequency is substantially lower.

**Important:** this is mechanics/behavior evidence, not proof that the strategy is strong. Self-play against itself cannot establish absolute chip-EV quality or exploitability. Do not authorize more training merely because the checkpoint exists, and do not call the policy final merely because self-play passed.

## Ryzen optimization is mandatory

`docs/RYZEN_OPTIMIZATION_POLICY.md` is canonical. This is also a standing user requirement beyond SpinCore: any substantial workload assigned to the user's Ryzen, in any project, must be optimized for that machine before long execution.

SpinCore now parallelizes independent advantage-root collection with a persistent process pool. Each worker uses one Torch/OpenMP/BLAS thread. The parent remains authoritative for empirical scenario sampling and global reservoirs and merges returned samples in root order. Poker rules, state representation, legal-action semantics, utility and intended Deep-CFR operation are unchanged by the execution optimization.

The parallel path already passed the dedicated GitHub lean functional benchmark with two workers. A broad legacy regression still has one stale frozen-blob-hash failure unrelated to this functional path; do not turn that certification-only hash into a workstream.

## Next milestone — ONE hardware tune, then ONE meaningful strategy decision

1. Run `tools/benchmark_lean_ryzen_workers.sh` once on the actual Ryzen. It compares root workers 1/8/16/24/31 and then parent Torch threads 1/2/4/8/16. Persist the fastest safe profile.
2. **Do not start another training run automatically.** First build/run one compact strategy-quality comparison for the completed checkpoint under the full empirical sampler using common deals and chip-EV-oriented evidence against a stable fixed baseline/opponent set.
3. Only if that evidence says the policy is still undertrained or strategically weak should the checkpoint be extended — and any extension must use the measured Ryzen-optimized profile.
4. If a concrete semantic/action/representation weakness appears, fix that weakness before spending more compute. Do not answer bad poker with blind extra training.

## Do not do

- Do not discard or overwrite the completed 120k checkpoint.
- Do not immediately extend it just because training completed.
- Do not use self-play PASS as proof of poker strength.
- Do not resume Dense-reference i3-i5 merely to complete an old matrix.
- Do not use fixed-10/20 PF0-PF4 evidence as the final global selector.
- Do not restart the old R8/gate chain.
- Do not create payout-specific trainings now.
- Do not reopen representation/action tournaments without concrete play evidence.
- Do not knowingly run long serial/low-utilization workloads on the Ryzen when independent work can be parallelized safely.
