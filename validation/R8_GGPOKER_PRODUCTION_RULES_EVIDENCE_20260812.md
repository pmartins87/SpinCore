# R8 GGPoker production-rules evidence — 2026-08-12

`R8.0 STATUS = INCOMPLETE / FAIL-CLOSED`

`READY FOR TABLES = NO`

This evidence record deliberately distinguishes **first-party facts proven today** from values that remain unresolved. No unresolved value may be filled from the R7.4 pilot, memory, a community table, or an approximation.

## First-party sources inspected

1. GGPoker current Spin & Gold rules/game page:
   - `https://ggpoker.com/pt-br/poker-games/spin-gold/`
   - current page observed 2026-08-12.
2. GGPoker current Spin & Gold strategy article:
   - `https://ggpoker.com/blog/ggpoker-spin-gold-strategy/`
   - published 2026-01-28; inspected 2026-08-12.
3. GGPoker first-party legal/localized Spin & Gold page used as a cross-check:
   - `https://legal.ggpoker.com/es/poker-games/spin-gold/`
   - inspected 2026-08-12.

## Proven current rules / structure facts

The current GGPoker Spin & Gold page proves:

- 3-Max is an offered Spin & Gold format.
- 3-Max buy-in selectors currently include USD $0.25, $1, $3, $5, $10, $20, $50, $100 and $200.
- The **starting chip stack, blind structure and time bank differ according to Spin Multiplier**.
- A tournament fee of 7% is withheld from each buy-in.
- One 52-card deck is used and shuffled after each hand.
- When two or more players are eliminated in the same hand, the player who started that hand with more chips ranks higher. If tied in starting chips, the player closer to the left of the dealer button ranks higher.
- Spin & Gold has no Make a Deal option.

The current first-party strategy article additionally states at a descriptive level:

- ordinary starting stacks are 500 chips but increase for larger multipliers;
- the blind structure is hyper-turbo and increases every three minutes;
- a 500-chip ordinary start corresponds to roughly 25 BB in the described standard early game;
- lower/standard multipliers are winner-take-all while higher multipliers can pay second and third place;
- the multiplier distribution presented in the strategy article is descriptive/approximate rather than a sufficient exact production table.

These descriptive statements are useful architectural evidence but **are not sufficient to materialize an exact production profile for every multiplier**.

## Dynamic-table binding cross-check — important fail-closed result

A second first-party cross-check exposed a concrete reason not to trust a dynamically rendered prize table unless the selected buy-in is explicitly bound to the extracted data.

The current main Spin & Gold page states that the USD $5 selector can reach a maximum multiplier/prize corresponding to **x200,000**. In contrast, the crawler-visible 3-Max table recovered from the first-party legal/localized page tops out at **x40,000** and contains rows including x40,000, x100, x50, x10, x5, x4, x3 and x2.

Those two first-party observations are not treated as contradictory game rules. They demonstrate that the retrieved legal-page table is a **dynamic selected-state artifact whose buy-in binding is missing from the extracted representation**. Therefore:

```text
A table that is internally complete is still NOT production evidence
unless its selected buy-in / jurisdiction / format state is proven.
```

In particular, the x40,000 table must **not** be silently assigned to the $5 profile merely because both came from official GGPoker pages. Doing so would create a valid-looking but semantically wrong production identity.

This cross-check strengthens the R8.0 fail-closed requirement: exact prize/frequency rows must be captured together with the state that selected them, preferably from the client, an exact first-party API/data payload, or an official static rule document that names the buy-in explicitly.

## Explicitly unresolved — must be proven before R8.0 PASS

The current public page exposes dynamic controls/tables whose complete selected-buy-in values were not reliably recoverable in this audit. The following remain unresolved and therefore cannot yet be frozen as production constants:

- exact starting-chip stack for each 3-Max Spin Multiplier;
- exact complete blind sequence for each multiplier;
- exact time-bank rule for each multiplier;
- exact multiplier set available for each buy-in;
- exact payout vector and frequency for every buy-in × multiplier combination;
- an explicit binding between every recovered dynamic table and its selected buy-in / 3-Max state;
- whether any jurisdiction/client configuration relevant to the intended production account differs from the global/current public presentation.

A search-rendered first-party localized page exposed one complete prize table, but because the dynamic page has multiple buy-in selectors and the retrieved representation does not cryptographically or semantically bind that table to a selected buy-in state, **that table is not promoted here into a production profile**.

## Consequence for architecture

Production policy identity must include at least:

```text
buy-in
× Spin Multiplier
× starting stack
× full blind structure
× normalized payout shares
× ruleset
× action abstraction
× utility model
× learning profile
× strategy domain (HU or 3H)
```

The code-level `SPINCORE_R8_PRODUCTION_PROFILE_V3` binds `currency`, `buy_in_minor_units`, multiplier, stack, blind levels, normalized payout shares and the remaining strategy identities. A change in any semantic field produces a new `spinprofile-v3:` identity; HU and 3H then derive separate policy IDs from that profile.

`ProductionEvidence` is now fail-closed at the schema level rather than relying on a free-text note. Every evidence record must declare a scope and the fields it proves. Facts that can vary with the selected game state use `SELECTED_PROFILE_STATE` and must carry exact bindings for:

```text
table_size
buy_in_minor_units
multiplier
```

A `GLOBAL_GAME` record is forbidden from claiming state-dependent fields such as starting stack, blind levels or payout shares. `ProductionProfile` rejects a selected-state evidence record whose binding differs from its own table/buy-in/multiplier and rejects profiles whose evidence set does not cover all required economic/structural fields.

This closes the specific architecture hole exposed by the dynamic-table audit: **an official GGPoker URL alone can no longer validate a profile-dependent table.**

Implementation/regression evidence:

```text
python/spincore/production_profile.py
schema = SPINCORE_R8_PRODUCTION_PROFILE_V3
commit = e13ab862909577c1c22b9be5f59c3e4e3916a253
binding tests commit = 42aec7a4c39c1da376c39a9863465da5b39a4573
main regression run = 31638697150
main regression = PASS
```

This validation is deliberately semantic/provenance validation; it cannot make an incorrect human transcription true. The eventual R8.0 evidence package must therefore preserve the actual first-party capture or exact first-party data payload from which the bound constants were transcribed.

## R7.4 pilot constants are not production evidence

The R7.4 validation configuration using values such as total 1500 chips, 10/20 and payout `(0.5, 0.3, 0.2)` remains a **pilot/test configuration only**. It must not become an R8 production profile unless the exact target buy-in/multiplier structure is independently proven to match it.

## Required next evidence

R8.0 can close only after the missing multiplier-specific structure is recovered from a first-party source. Preferred evidence order:

1. exact GGPoker client rules/structure captured for the production jurisdiction/account, including the selected buy-in/multiplier state;
2. exact first-party web/API data behind the dynamic table, with selected-state binding preserved in the same capture;
3. an official GGPoker rule document/help record that enumerates the same fields and explicitly names the profile to which they apply.

Until then:

```text
R8.0 production profile/rules = INCOMPLETE
R8.1 infrastructure engineering = MAY CONTINUE
R8.2 calibration = BLOCKED
R8.3/R8.4 official training = BLOCKED
READY FOR TABLES = NO
```
