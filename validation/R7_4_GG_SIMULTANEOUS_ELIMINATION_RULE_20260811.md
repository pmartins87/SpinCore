# R7.4 ruleset extension — GGPoker same-hand tournament eliminations

Date: 2026-08-11  
Ruleset schema after extension: `SPINRULESET-4`  
`READY FOR TABLES = NO`.

## Triggering physical failure

The first deliberately added three-handed partial-exact physical smoke reached a terminal state that the recovered Generation-2 tournament utility could not value:

```text
workflow: 31471334618
C++ regression: PASS
Python regression: 1 failed / 65 passed
failing test: test_r7_4_three_handed_partial_exact_collection_physically_executes
runtime error: ambiguous simultaneous elimination under unequal payouts
```

The failure was not suppressed. It exposed a real production-rules gap: the old context-free continuation function knew the starting stacks and final stacks but did not know table/button position, so equal-starting-stack players eliminated in the same hand could not be ranked when the next two payouts differed.

## Authoritative GGPoker rule

Primary sources checked before changing engine semantics:

- GGPoker House Rules — Tournament Elimination Policy: https://legal.ggpoker.com/house-rules/
- GGPoker Spin & Gold rules/details: https://ggpoker.com/poker-games/spin-gold/

Both state the same ordering rule. When multiple players are eliminated on the same hand:

1. the player who started the hand with more chips ranks higher;
2. if their starting chip counts are equal, table position breaks the tie;
3. the eliminated player closest to / first seated to the left of the dealer button ranks higher than eliminated players farther away.

This exactly fills the information gap because `SpinTraversalState` already owns `EpisodeScenario.dealer_id`.

## `SPINRULESET-4` implementation

The extension is intentionally narrow.

### Context-free API remains fail-closed

The existing overload remains:

```cpp
terminal_continuation_delta(before, final_stacks, payout)
```

If it encounters equal-starting-stack simultaneous eliminations under unequal payouts, it still throws. This preserves the previous safety contract because that API has no dealer/button context and must not invent a ranking.

### Traversal gets a dealer-aware overload

New overload:

```cpp
terminal_continuation_delta(before, final_stacks, payout, dealer_id)
```

`SpinTraversalState::terminal_icm_delta()` now supplies `scenario_.dealer_id`.

For equal-starting-stack simultaneous eliminations, seats are ordered cyclically beginning immediately left of the button. Because dead-player history is stored worst-place first, the farther eliminated seat is appended before the nearer-left-of-button seat.

For unequal starting stacks, the existing ordering is unchanged: the player who started the hand with more chips finishes ahead.

## Frozen implementation lineage

The ruleset-extension implementation is complete at descendant commit:

```text
e43b2cfea31f927393cf2751485d712902d6f02d
```

Core commits:

```text
6257b40ae5b8cd62b1b2046ac1e62c3e06f2be35  dealer-aware continuation API
8d2d1c957d6295d0c19f8cf1878c931e17798b66  positional tie resolution
62885416d841e9bb43d9a25d85b1b905bf09b835  traversal passes dealer_id
e43b2cfea31f927393cf2751485d712902d6f02d  tournament-value rule tests
```

Changed strategic/core files for this extension:

```text
include/spincore/tournament_value.hpp
src/tournament_value.cpp
src/spin_traversal_state.cpp
```

The selected R7.3 uncertainty runner and its important helper runners are byte-identical between the selected R7.3 source and the extension source; R7.4 still must prove physical HU invariance before this descendant may become the R7.4 strategic source.

## Physical regression after the extension

Workflow `31471689918`, head `e43b2cfea31f927393cf2751485d712902d6f02d`:

```text
C++ regression: PASS
Python regression: 66 passed
```

Most importantly, the same physical three-handed partial-exact smoke that had failed now completes. This establishes that 3H traversal can cross the previously ambiguous terminal state under the dealer-aware production rule.

## C++ rule coverage

The regression suite now verifies:

- context-free equal-stack/unequal-payout ambiguity still fails closed;
- dealer 0 positional ordering;
- dealer 1 positional ordering;
- dealer 2 positional ordering;
- invalid dealer rejection;
- continuation deltas remain zero-sum;
- unequal-starting-stack simultaneous elimination preserves the old rule.

## Effect on R7.3 certification

This extension does **not** rewrite or retroactively invalidate the selected R7.3 source. R7.3 certification remains pinned to its immutable HU source and evidence.

The new rule is needed for three-handed R7.4. Before any R7.4 strategic held-out result is accepted, SpinCore must prove that the `SPINRULESET-4` descendant preserves the selected R7.3 true-HU behavior under the exact selected schedule. Only then is the descendant frozen as the R7.4 rules source.

Therefore the finite R7.4 sequence becomes:

```text
corrected R7.3 certification PASS
-> SPINRULESET-4 extension freeze
-> physical selected-winner HU invariance PASS
-> R7.4 structural preflight on frozen SPINRULESET-4 source
-> precommitted held-out HU/3H gate
```

No R7.4 threshold, held-out seed rule, scenario set, or scale is changed by this rules correction.
