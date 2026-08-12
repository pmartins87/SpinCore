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
- lower/standard multipliers are winner-take-all while higher multipliers can pay second and third place.

These descriptive statements are useful architectural evidence but **are not sufficient to materialize an exact production profile for every multiplier**.

## Explicitly unresolved — must be proven before R8.0 PASS

The current public page exposes dynamic controls/tables whose complete selected-buy-in values were not reliably recoverable in this audit. The following remain unresolved and therefore cannot yet be frozen as production constants:

- exact starting-chip stack for each 3-Max Spin Multiplier;
- exact complete blind sequence for each multiplier;
- exact time-bank rule for each multiplier;
- exact multiplier set available for each buy-in;
- exact payout vector and frequency for every buy-in × multiplier combination;
- whether any jurisdiction/client configuration relevant to the intended production account differs from the global/current public presentation.

A search-rendered first-party localized page exposed one complete prize table, but because the dynamic page has multiple buy-in selectors and the retrieved representation does not cryptographically bind that table to a selected buy-in state, **that table is not promoted here into a production profile**.

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

The code-level `SPINCORE_R8_PRODUCTION_PROFILE_V2` therefore binds `currency`, `buy_in_minor_units`, multiplier, stack, blind levels, normalized payout shares and the remaining strategy identities. A change in any semantic field produces a new profile hash; HU and 3H then derive separate policy IDs from that profile.

## R7.4 pilot constants are not production evidence

The R7.4 validation configuration using values such as total 1500 chips, 10/20 and payout `(0.5, 0.3, 0.2)` remains a **pilot/test configuration only**. It must not become an R8 production profile unless the exact target buy-in/multiplier structure is independently proven to match it.

## Required next evidence

R8.0 can close only after the missing multiplier-specific structure is recovered from a first-party source. Preferred evidence order:

1. exact GGPoker client rules/structure captured for the production jurisdiction/account;
2. exact first-party web/API data behind the dynamic table, if recoverable and bindable to the selected buy-in;
3. an official GGPoker rule document/help record that enumerates the same fields.

Until then:

```text
R8.0 production profile/rules = INCOMPLETE
R8.1 infrastructure engineering = MAY CONTINUE
R8.2 calibration = BLOCKED
R8.3/R8.4 official training = BLOCKED
READY FOR TABLES = NO
```
