# DeepCrusher vs CrusherTBP — Benchmark Contract

Date: 2026-10-01  
Status: **PREREGISTRATION DRAFT — FAIL CLOSED BEFORE CANONICAL RUN**

This contract applies only to DeepCrusher-vs-CrusherTBP. It does not modify the historical SpinCore-vs-DeepCrusher benchmark contract.

## G0 — immutable source manifest

Before parity or EV evaluation, record for each policy:

- repository commit and branch/tag;
- exact strategy filename, size and SHA256;
- exact OpenPPL library files and SHA256;
- OpenHoldem source/runtime reference and hashes;
- external files, databases, aliases, notes, PT state, user variables and other runtime dependencies;
- frozen environment profile;
- benchmark-engine commit.

Any source/hash change creates a different benchmark candidate and requires a new manifest.

**Current gate: BLOCKED.**

## G1 — transitive semantic closure

Roots include `f$preflop`, `f$flop`, `f$turn`, `f$river` and every reserved lifecycle/autoplayer callback reachable for that policy.

Every reachable dependency must resolve to one of:

- strategy section;
- pinned OpenPPL library section;
- exact native OpenHoldem symbol implementation;
- explicit environment value;
- exact hand-list/list implementation;
- exact state/history/memory implementation.

Unknown or approximated strategic symbols are hard failures. No silent zero-fill.

## G2 — exact OpenHoldem action pipeline

Each policy keeps its own action semantics and sizing. The simulator must preserve at least:

- positive f$betsize;
- negative percentage-pot f$betsize;
- fixed OpenPPL action codes;
- Fold / Check / Call / Bet / Raise;
- RaiseTo and RaiseBy;
- exact technical pot fractions;
- `AmountToCall`, `BetSize`, current bets, pot and ncallbets;
- minimum legal raise, maximum legal raise and all-in cap;
- `f$allin_on_betsize_balance_ratio` on every OpenHoldem path where it is evaluated;
- OpenPPL backup/legal fallback cascade;
- deterministic chip rounding only where the target simulator requires integer chips.

No policy may be projected into the other policy's betting abstraction.

**Unexplained action-family or amount mismatch vs real OpenHoldem = HARD FAIL.**

## G3 — state and lifecycle

Freeze and test:

- chair/dealer/position topology;
- blinds, stacks, balances, current bets and total commitments;
- pot/common-pot semantics;
- board/hole cards and exact suits;
- opponent action transcript and history symbols;
- `user_` state lifetime;
- `me_st_`, `me_re_` and related memory lifetimes;
- connection/startup/hand-reset/new-round/my-turn callbacks in actual OpenHoldem order;
- street transitions and cleanup;
- no state leakage between paired replays.

## G4 — real OpenHoldem parity for both policies

Build deterministic fixtures from real OpenHoldem covering both DeepCrusher and CrusherTBP.

Required fixture families:

- HU and 3H;
- preflop, flop, turn and river;
- fold/check/call;
- ordinary bet/raise;
- RaiseTo / RaiseBy;
- 1/4, 1/3, 1/2, 2/3, 3/4 and pot sizings where reachable;
- minraise and maxraise;
- all-in direct and all-in created by commitment adjustment;
- legal-action fallback;
- representative history/state-memory branches;
- proposed no-PT/no-note environment.

Parity compares the final executed action and amount, plus relevant state/memory side effects.

No aggregate EV simulation beyond tiny fixture-driving smoke is admitted before G4 PASS for both policies.

## G5 — paired mechanical benchmark

Only after G4 PASS:

### HU

For each sampled state/deal, play the same complete deal twice while swapping DeepCrusher and CrusherTBP across the two live seats.

### 3H

For each sampled state/deal, use a six-game AAB/ABB block:

- three games with one DeepCrusher seat rotating across all logical seats;
- three games with one CrusherTBP seat rotating across all logical seats.

Each policy must receive exactly equal total exposure and equal exposure by logical seat.

### Hard invariants

- same cards/deal inside the paired block;
- same stacks, blinds, payout/context and starting public state;
- every terminal game exactly zero-sum in chips;
- every complete block exposure-balanced;
- policy aggregate exactly zero-sum;
- any illegal action, impossible state, untranslated action, source/hash mismatch or exposure mismatch aborts the run.

## G6 — worker-count equivalence and performance

Before large scale:

- run the same deterministic benchmark slice with 1 worker and each candidate worker layout;
- require identical scenario/deal identities and exact terminal results/traces modulo ordering/metadata that is intentionally nondeterministic;
- record wall time, peak RSS/RAM and swap;
- workers load only compact runtime assets and use one BLAS/OpenMP/Torch thread unless exact-equivalence testing proves another configuration safe;
- select the fastest exact-equivalent layout before the canonical run.

Performance is subordinate to parity.

## G7 — preregistered primary statistic and canonical evidence

Primary sign convention:

`delta = EV_DeepCrusher - EV_CrusherTBP`

Unit of analysis: **paired scenario cluster**, not individual game row.

For each cluster, aggregate policy chip results after the full exposure-balanced block and normalize by equal policy-seat-hand exposure.

Report:

- mean paired delta in chips/policy-seat-hand;
- 95% confidence interval using the preregistered scenario-cluster estimator;
- overall;
- THREE_HANDED separately;
- TRUE_HEADS_UP separately.

Development, fixture and canonical seed namespaces must be disjoint.

Initial canonical target: **at least 100,000 sampled scenario clusters**, provided worker-equivalence and resource gates are already PASS.

If extension is allowed, its chunk size and stopping rule must be frozen before reading the canonical result and depend only on required confidence-interval precision / maximum budget, never on which policy is ahead.

No threshold or decision criterion may be changed after observing the canonical result.

## G8 — retained forensic evidence

Persist a manifest and enough full decision traces / deterministic replay keys to audit:

- monster folds;
- absurd jams;
- impossible calls;
- wrong sizing;
- illegal actions;
- wrong position/chair;
- bad pot/current-bet/stack;
- routing mistakes;
- state contamination;
- simulator-vs-OpenHoldem disagreement.

Every row must identify scenario, deal/replay key, lineup, seat, street, policy, cards available to that actor, public state, selected OpenPPL provenance, final exact action and final amount.

## G9 — preregistered vs exploratory decomposition

The canonical primary result is frozen before subgroup exploration.

After the primary analysis, decomposition may examine:

- HU vs 3H;
- stack depth;
- position/logical seat;
- preflop/flop/turn/river;
- SRP / limped / 3bet+ families where reconstructable;
- hand strength/draw class;
- fold/call/raise mix;
- aggression and sizing;
- largest per-spot EV deltas;
- pathology frequencies.

Any breakdown not explicitly preregistered before the canonical run is labelled **EXPLORATORY / POST HOC** and may generate a future hypothesis but not retroactively redefine the canonical conclusion.

## Current stop condition

Do not start a large benchmark.

Next admissible work is G0 -> G4: exact source freeze, semantic closure, action-pipeline correction, lifecycle parity and real OpenHoldem fixtures for both policies.
