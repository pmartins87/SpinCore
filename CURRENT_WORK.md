# SpinCore Current Work

Date: 2026-09-15
Status: **FIRST SUBSTANTIVE TRAINING COMPLETED — RYZEN PROFILE MEASURED (31 WORKERS / 8 PARENT THREADS) — STRATEGY EVAL RUNNING; DEEPCRUSHER BENCHMARK UNDER CONSTRUCTION**

## Goal

Make the multi-year DeepSpin project actually work as SpinCore. Preserve mature legacy knowledge; replace only components with a concrete correctness, learning-quality, or compute-efficiency reason.

A product-quality acceptance metric is now explicit: **SpinCore must beat DeepCrusher in extensive fair offline simulation under common game semantics.** DeepCrusher is a mandatory reference opponent, not the sole training teacher and not the definition of optimal play. The benchmark must use common deals where possible, seat rotation, the realistic blind/stack distribution, enough volume to suppress card variance, and poker outcomes reported overall and by HU/3H/blind/position. Direct HU is the cleanest first head-to-head comparison; 3H must use balanced/mirrored lineups. A future continuous-tournament simulator should add full-match/tournament win rate once the tournament transition schedule is explicitly defined.

Poker-level explanation of what SpinCore is learning is canonical in `docs/POKER_GOALS_AND_TRAINING_EXPLAINED.md`. The direct head-to-head benchmark contract is canonical in `docs/DEEPC_RUSHER_BENCHMARK_SPEC.md`.

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

## Ryzen optimization — COMPLETED 2026-09-15

`docs/RYZEN_OPTIMIZATION_POLICY.md` is canonical. This is also a standing user requirement beyond SpinCore: any substantial workload assigned to the user's Ryzen, in any project, must be optimized for that machine before long execution.

The measured one-time hardware benchmark completed successfully on the actual 32-logical-thread Ryzen.

Phase 1 — advantage-root workers:

- 1 worker: 33.379956 s tree time;
- 8 workers: 5.795283 s;
- 16 workers: 6.598084 s;
- 24 workers: 3.832261 s;
- **31 workers: 3.678355 s — selected**.

Phase 2 — parent Torch threads while keeping 31 root workers:

- 1 thread: 3.524190 s steady iteration;
- 2 threads: 3.012554 s;
- 4 threads: 2.898485 s;
- **8 threads: 2.560276 s — selected**;
- 16 threads: 3.543747 s.

All 5 worker candidates and all 5 parent-thread candidates completed successfully. The persistent production profile is therefore:

- **root workers = 31**;
- **parent Torch threads = 8**;
- worker Torch/OpenMP/BLAS threads remain 1 each.

The selected values are stored locally in:

- `runs/worker_benchmark/selected_workers.txt`;
- `runs/worker_benchmark/selected_torch_threads.txt`.

Do not infer that the whole training job becomes 13x faster merely from the inner benchmark. Root collection accelerated dramatically, but optimizer/final-policy work and process/model startup remain partly serial or fixed-cost. Any future long-run ETA must be measured from the optimized substantive path itself.

## Strategy-quality evaluation now prepared/running

Before any further training, run exactly one paired full-sampler chip-EV diagnostic:

- tool: `tools/evaluate_lean_strategy_quality.py`;
- one-command wrapper: `tools/run_lean_strategy_quality_eval.sh`;
- default workload: 1,000 empirical scenarios under the full 3H/HU blind-conditioned sampler;
- rotate the evaluated hero through every live seat;
- compare the learned SpinCore AveragePolicy against a uniform-legal hero control on the **same scenario and deal**;
- fixed opponent families: `UNIFORM_LEGAL`, `PASSIVE_CALLER`, `JAMMER`;
- primary evidence: raw hero chip EV plus paired `SpinCore - uniform-control` chip-EV gain, with 95% CI clustered by scenario;
- report overall and separately for 3H, HU and blind-level detail;
- use the measured worker count automatically, with one Torch/OMP/MKL thread per evaluation worker to avoid oversubscription.

This is a diagnostic against transparent fixed weak baselines, **not** an exploitability/GTO proof and not a replacement for the mandatory DeepCrusher benchmark. Its purpose is to answer a concrete pre-training question: did the learned policy acquire measurable strategic value beyond an untrained legal-action policy, and does any domain show a gross red flag? If it fails this basic comparison, inspect/fix the cause before spending more compute. If it passes, advance to stronger poker-specific evaluation, including direct DeepCrusher head-to-head rather than blindly extending training.

## Mandatory DeepCrusher acceptance benchmark — BUILDING NOW

Canonical specification: `docs/DEEPC_RUSHER_BENCHMARK_SPEC.md`.

Frozen first opponent is the DeepCrusher R8 v22 good/stable baseline from `pmartins87/DeepCrusher`, branch `r8-v22-stable-20260914`:

- operational OpenHoldem file SHA256 `0113badc99727a7dd47c02448d4d042b5e008534cd63fd79a461a72b24eeb68d`;
- strategic recovered file SHA256 `9fc2d00aacc915f3c265429f764056f3c6270df616244026aac22e455c803ee9`.

Already implemented in SpinCore:

- exact source/hash pin and source-preflight contract;
- additive solver API to apply an external policy's exact Fold/Check/Call/BetTo/RaiseTo/AllIn action without quantizing DeepCrusher into SpinCore sizes;
- HU paired same-deal seat swap;
- 3H six-game AAB/ABB balanced block: each strategy receives exactly nine player-seat exposures and exactly three exposures in every logical seat per sampled state/deal;
- zero-sum strategy aggregation;
- unit tests for HU/3H pairing and exact-action bridge;
- OpenPPL section/dependency inventory for the frozen DeepCrusher source.

The main remaining implementation is the **DeepCrusher decision oracle**: given one exact offline SpinCore simulator state, it must return the same action and exact bet size as the frozen OpenPPL strategy. Do not shortcut this by inventing a simplified imitation. The oracle must be validated against real OpenHoldem reference decisions before an extensive result can be called canonical.

Benchmark stages are frozen as:

1. DC0 source/runtime parity for the DeepCrusher oracle;
2. DC1 1k–5k paired-state smoke for mechanics/fairness;
3. DC2 extensive paired empirical-sampler chip-EV benchmark, initially >=100k states and then adaptive CI stopping;
4. DC3 complete Spin & Go tournament win rate after continuous blind/dealer/elimination progression is explicitly frozen.

Primary DC2 acceptance: overall `SpinCore - DeepCrusher` chip EV > 0 with 95% CI lower bound > 0, without the result being carried by one isolated blind/position while another major domain collapses. DeepCrusher remains a reference opponent, not the sole definition of optimal play.

## Do not do

- Do not discard or overwrite the completed 120k checkpoint.
- Do not immediately extend it just because training completed.
- Do not use self-play PASS as proof of poker strength.
- Do not use an approximate/caricature DeepCrusher oracle for a canonical head-to-head claim.
- Do not quantize DeepCrusher bet sizes into SpinCore's seven-action abstraction during the head-to-head.
- Do not resume Dense-reference i3-i5 merely to complete an old matrix.
- Do not use fixed-10/20 PF0-PF4 evidence as the final global selector.
- Do not restart the old R8/gate chain.
- Do not create payout-specific trainings now.
- Do not reopen representation/action tournaments without concrete play evidence.
- Do not knowingly run long serial/low-utilization workloads on the Ryzen when independent work can be parallelized safely.

## Immediate next milestone

Let the currently running `tools/run_lean_strategy_quality_eval.sh` finish; do **not** start another training run. In parallel, continue implementing the DeepCrusher oracle and validate the new exact-action/pairing infrastructure in CI. Once the strategy sanity result is available, interpret it while continuing DC0 rather than waiting to start benchmark construction.
