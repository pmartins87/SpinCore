# R8.0 — GGPoker official web evidence, 2026-08-12

`R8.0 PASS = NO`

`READY FOR TABLES = NO`

This record captures only facts directly supported by current first-party GGPoker web material observed on 2026-08-12. It intentionally does not infer selected-profile stack/blind/payout values from generic prose, validation pilot constants, old blog posts, or a dynamically rendered default table.

## First-party sources observed

- GGPoker Brazil Spin & Gold game page: `https://ggpoker.com/pt-br/poker-games/spin-gold/`
- GGPoker legal/global Spin & Gold page (Spanish localization exposing the currently indexed default probability table): `https://legal.ggpoker.com/es/poker-games/spin-gold/`

## Facts supported globally

The current official Spin & Gold page states:

- Spin & Gold is offered in 3-Max and 6-Max forms.
- Current 3-Max buy-in buttons shown by the official page are USD 0.25, 1, 3, 5, 10, 20, 50, 100 and 200.
- The page advertises wins up to 200,000x and specifically states that a USD 5 buy-in can reach USD 1,000,000.
- A tournament fee of 7% is withheld from each buy-in.
- The game uses one 52-card deck and the deck is shuffled after every hand.
- Starting chips, blind structure and time bank vary according to the Spin multiplier.
- If two or more players are eliminated in the same hand, the player who started that hand with more chips ranks higher. If the relevant players started the hand with equal chips, the player closer to the left of the dealer button ranks higher.
- Spin & Gold does not offer Make a Deal.

These global facts may be used as provenance only where the `SPINCORE_R8_PRODUCTION_PROFILE_V3` evidence model permits global evidence. In particular, the 7% tournament fee is a global production-profile fact supported by the official page.

## Dynamic selected-state data currently NOT proven

The public page explicitly says that starting chips and blind structure vary with the Spin multiplier, but the text representation available to the current evidence collector does not expose a complete selected-state mapping for:

- exact `table_size × buy_in × multiplier` starting chips;
- exact full blind level sequence and level duration for each selected state;
- exact payout vector for every 3-Max buy-in/multiplier state.

The indexed Spanish page exposes one currently rendered/default 3-Max probability/payout table, including x2/x3/x4/x5/x10/x50/x100/x40,000 rows. That table is **not** promoted to every buy-in or multiplier profile because the page itself is dynamic and the selected buy-in binding is not captured in the retrieved representation.

Likewise, an official 2026 strategy article saying that games generally start with 500 chips is informative but is not accepted as selected-profile evidence because the official game page explicitly states that starting chips depend on the multiplier.

## R8.0 consequence

The production-profile contract remains fail-closed. No R7.4 pilot constants are promoted into R8, and no profile is materialized until first-party evidence is explicitly bound to the exact 3-Max selected state and proves all mandatory V3 fields:

- table size;
- buy-in;
- multiplier;
- starting chips;
- blind levels;
- payout shares;
- tournament fee.

Current progress therefore reduces the unknown surface but does not satisfy R8.0. The remaining evidence acquisition problem is specifically the dynamic selected-state capture for stack/blind/payout data.
