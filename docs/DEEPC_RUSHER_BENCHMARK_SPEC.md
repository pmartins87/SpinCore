# SpinCore vs DeepCrusher — Benchmark Specification v1

Status: **ACTIVE / BUILDING**
Date: 2026-09-15

## Objective

One of SpinCore's mandatory quality goals is to **beat the frozen DeepCrusher baseline in extensive offline poker simulation**.

This benchmark is not allowed to become a cosmetic test.  It must answer a poker question:

> When both strategies receive the same realistic Spin & Go situations and the same cards, which strategy wins more chips, and later which strategy wins more complete tournaments?

The benchmark is also not the training target.  DeepCrusher is a strong reference opponent; SpinCore must not be trained merely to exploit DeepCrusher-specific leaks.

## Frozen DeepCrusher opponent

Initial canonical opponent:

- branch: `r8-v22-stable-20260914`;
- strategic source: `DeepCrusher_R8_v22_CANDIDATE_OPENHOLDEM_RECOVERED_20260914.txt`;
- strategic SHA256: `9fc2d00aacc915f3c265429f764056f3c6270df616244026aac22e455c803ee9`;
- operational OpenHoldem source: `DeepCrusher_R8_v22_CANDIDATE_OPENHOLDEM_ASCII_20260914.txt`;
- operational SHA256: `0113badc99727a7dd47c02448d4d042b5e008534cd63fd79a461a72b24eeb68d`.

This is the DeepCrusher repository's frozen good/stable R8 v22 baseline.  R9 may later become a second benchmark opponent once it is itself frozen; it must not silently replace the historical R8 v22 result.

## Poker fairness principles

### Same situation, same cards

SpinCore and DeepCrusher must be compared on paired deals.  A lucky distribution of AA/KK to one strategy cannot determine the result.  Whenever practical, the same sampled stack/blind/dealer state and the same complete deal are reused while policy assignments are swapped.

### Keep each strategy's own bet sizes

DeepCrusher must not be translated into SpinCore's action abstraction.  If DeepCrusher chooses its own exact bet/raise size, the simulator applies that exact size.  Conversely SpinCore continues to use the sizes its policy was trained with.

The SpinCore solver therefore exposes an additive exact-action benchmark API for Fold, Check, Call, BetTo, RaiseTo and AllIn.

### Realistic Spin & Go state distribution

The hand-level benchmark uses the same empirical 3H/HU blind-conditioned sampler used by SpinCore training:

- total chips 1500;
- 3H/HU mixture and stack distributions from the recovered DeepSpin data;
- blinds from 10/20 through 100/200 according to their observed frequencies;
- random dealer/live seats consistent with the domain.

The benchmark does not cherry-pick only 10/20 or only comfortable stack depths.

## HU pairing

For one sampled HU state/deal there are two benchmark games:

1. SpinCore occupies live seat A and DeepCrusher live seat B;
2. the policies swap seats on the same state/deal.

Across a large sample each policy therefore receives the same card and position opportunities.

Primary HU metrics:

- SpinCore minus DeepCrusher chips/hand;
- chips/hand confidence interval;
- bb/100 within each blind level;
- BTN/SB and BB split;
- preflop/flop/turn/river contribution;
- all-in frequency, VPIP/PFR/3-bet-style action families where recoverable from exact actions.

## 3-handed pairing

A naïve test `SpinCore + DeepCrusher + DeepCrusher` is biased because one policy always appears once and the other twice.

For every sampled 3H state/deal the canonical block has **six games**:

- three `SpinCore/SpinCore/DeepCrusher` games, rotating the single DeepCrusher through all three seats;
- three `SpinCore/DeepCrusher/DeepCrusher` games, rotating the single SpinCore through all three seats.

Across the six-game block:

- SpinCore has exactly 9 player-seat exposures;
- DeepCrusher has exactly 9 player-seat exposures;
- each policy occupies each logical seat exactly 3 times;
- majority/minority exposure is symmetric;
- all chip deltas remain globally zero-sum.

This gives a fair population-vs-population comparison without pretending that a three-seat table can contain equal counts of two strategies in one single hand.

## Benchmark stages

### Stage DC0 — source/runtime fidelity

Before trusting any score, the DeepCrusher decision adapter must be proven to represent the actual frozen OpenPPL strategy.

Required evidence:

- exact source filename/hash pin passes;
- DeepCrusher decision adapter reproduces a broad reference trace from the real OpenHoldem runtime;
- preflop and every postflop street are represented;
- exact bet/raise amount parity is checked, not only action names;
- topology distinction native HU vs 3H-origin heads-up is preserved.

A benchmark using an approximate imitation of DeepCrusher may be useful for development but cannot be called the canonical SpinCore-vs-DeepCrusher result.

### Stage DC1 — paired hand-level chip EV smoke

Purpose: mechanical validation only.

Suggested size: 1,000–5,000 sampled states.

Checks:

- no illegal actions;
- no state divergence;
- same-deal pairing works;
- HU seat swaps balance exactly;
- 3H six-game blocks balance exactly;
- terminal chip accounting is zero-sum;
- exact DeepCrusher sizing is retained.

No quality conclusion from DC1.

### Stage DC2 — qualification hand-level benchmark

Purpose: establish whether SpinCore has a statistically meaningful edge over DeepCrusher under the empirical hand-state distribution.

Start with at least **100,000 sampled states**, executed in Ryzen-optimized parallel form.  Continue in chunks only when the confidence interval is still too wide; do not spend compute merely to hit a ceremonial round number.

Primary metric:

- **paired SpinCore minus DeepCrusher chip EV, chips/hand**.

Secondary metrics:

- HU and 3H separately;
- bb/100 by blind level;
- BTN/SB/BB or native HU position;
- stack-depth bands;
- preflop and postflop contribution;
- blind-level and position red flags.

Qualification success requires:

1. overall SpinCore-minus-DeepCrusher chip EV > 0;
2. 95% CI lower bound > 0 for the overall paired result;
3. the edge is not explained by one isolated blind level or one position while another major domain collapses;
4. no material illegal-action or action-translation rate.

The exact precision stop rule can be tightened after the DC1 variance is observed.  Useful precision, not sample-count vanity, controls runtime.

### Stage DC3 — full Spin & Go tournament benchmark

This is the user's strongest intuitive target: **SpinCore should win more complete Spin & Gos than DeepCrusher**.

DC3 will start only after continuous tournament semantics are separately frozen.  The current empirical sampler accurately represents hand states but does not itself define a sequential blind clock from hand 1 until elimination.

Before DC3 we must freeze:

- starting stacks;
- dealer movement;
- blind progression / hands or time per level;
- transition from 3H to HU;
- elimination/tie semantics;
- card-seed pairing between mirrored tournament runs.

Then run complete tournaments with balanced policy compositions and mirrored seats/deals where possible.

Primary tournament metric:

- **SpinCore tournament win rate versus DeepCrusher**, with 95% CI.

For pure HU complete matches, a direct target is win rate > 50% with CI lower bound > 50%.

For 3H mixed-population tournaments, report per-policy win share under the same balanced AAB/ABB block design rather than comparing raw seat totals from an unbalanced composition.

## DeepCrusher decision-adapter architecture

The benchmark has two independent pieces:

1. **match engine** — already being built in SpinCore and responsible for paired states/deals, lineups, exact action application and scoring;
2. **DeepCrusher oracle** — converts one exact simulator state into the same decision the frozen OpenPPL formula would make.

The oracle is the difficult part because DeepCrusher is a 1.27 MB OpenPPL/OpenHoldem strategy with many native OpenHoldem symbols.  We will not replace it with a simplified hand-written caricature merely to obtain a number quickly.

Preferred path:

- reproduce the OpenPPL symbol/action semantics needed by the frozen strategy in an offline portable adapter;
- validate every relevant family against captured real OpenHoldem decisions;
- only after parity call DC2 canonical.

A real OpenHoldem-backed offline oracle is also acceptable if it can consume deterministic simulated states without depending on a live poker client.

## Ryzen execution

Every extensive DC2/DC3 run is a substantial Ryzen workload and therefore follows `docs/RYZEN_OPTIMIZATION_POLICY.md`.

Current measured machine profile is 31 independent workers / 8 parent Torch threads for SpinCore training.  The benchmark has a different workload, so it may reuse 31 workers initially but must measure benchmark throughput and memory before a very large run.  Do not assume the trainer's exact worker optimum if the DeepCrusher oracle changes the bottleneck.

## Result preservation

Every canonical run must record:

- SpinCore checkpoint SHA/path/hash;
- DeepCrusher source filename/SHA256/branch;
- SpinCore commit;
- benchmark schema version;
- sampler seed(s);
- number of paired states / tournament blocks;
- worker profile;
- raw aggregate report and domain/blind/position breakdown;
- confidence interval method;
- whether DeepCrusher oracle parity was canonical or development-only.

No later SpinCore version may overwrite an older result.  Benchmark history is cumulative so regressions remain visible.

## Current build state

Implemented now:

- frozen DeepCrusher R8 v22 source/hash contract and source/dependency preflight;
- exact external BetTo/RaiseTo action application in the SpinCore solver;
- HU same-deal seat-swap schedule;
- fair 3H six-game AAB/ABB schedule;
- zero-sum per-policy aggregation contract;
- strict OpenHoldem-compatible expression evaluator;
- ordered WHEN / RETURN / SET structural evaluator with multiline source normalization;
- canonical parsing/evaluation of all list_* hand ranges;
- hand-scoped user_* variable persistence and connection-scoped me_* memory semantics derived from the preserved OpenHoldem implementation;
- full-source structural compile audit and static-preparation runner;
- CI contracts for the portable DC0 foundation.

Still required before DC0 can authorize canonical DC1/DC2:

- complete native/OpenPPL symbol projection from SpinCore observable state and action history;
- exact action/sizing translation, including the R8 f$BestBetsize and all-in-conversion paths;
- explicit frozen treatment of optional environment/PokerTracker/network-dependent symbols;
- broad real-OpenHoldem parity fixtures across preflop/flop/turn/river with exact bet/raise amounts.

The prior R8 v22 live smoke is useful external evidence that the frozen formula itself is operational, but its original five logs are not committed as machine-readable parity fixtures.


## Frozen offline environment profile

Canonical synthetic matches use `GGPoker_NoPT_NoNotes_V1` unless a later
profile is explicitly preregistered.

The profile is intended to remove non-strategic live-table identity/history
dependencies without silently zero-filling unknown strategy symbols:

- `network$ggpoker = 1`; other network$ symbols = 0;
- named `chair$...` lookups = OpenHoldem kUndefined (-1), representing no
  matching named player at the synthetic table;
- `log$...` = 1, matching OpenHoldem's expression semantics;
- colour-note symbols = 0 for synthetic unlabelled opponents;
- PokerTracker symbols = OpenHoldem kUndefined (-1), representing no PT
  connection/data.

`prwin` and `prtie` are NOT environment inputs. They are strategic equity
symbols and remain DC0 implementation dependencies.

## Decision trace and sanity-review contract

DC1/DC2 must retain machine-readable decision traces for SpinCore and
DeepCrusher. Aggregate EV is necessary but not sufficient for diagnosis.
SpinCore traces are additionally screened for obvious/high-value review cases
such as AA folds, non-trivially deep 72o jams, top-pair folds, trips+ folds and
deep high-card jams. Flags create a review queue and are not automatically
classified as strategy errors.
