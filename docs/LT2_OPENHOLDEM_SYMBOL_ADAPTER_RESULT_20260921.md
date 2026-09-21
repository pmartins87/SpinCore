# SpinCore — LT2 OpenHoldem symbol/scrape adapter result

Date: 2026-09-21  
Status: **PASS — STRICT OPENHOLDEM SYMBOL NORMALIZATION VALIDATED**

## Scope

The gate round-tripped authoritative SpinCore states through synthetic
OpenHoldem frames using the documented OpenHoldem symbol semantics.

No deployment-model inference, EV evaluation, optimizer work, training roots
or holdout reuse occurred.

## Coverage

- hand anchors: `1,271`;
- runtime frames: `9,000`;
- distinct physical chair layouts: `370`.

Domains:

- THREE_HANDED: `5,757`;
- TRUE_HEADS_UP: `3,243`.

Streets:

- preflop: `3,970`;
- flop: `2,440`;
- turn: `1,561`;
- river: `1,029`.

Hero logical positions:

- logical 0: `3,712`;
- logical 1: `3,316`;
- logical 2: `1,972`.

## Fault injection

Total malformed frames:
- attempts `1,200`;
- rejected `1,200`.

Breakdown:
- wrong hand id: 300/300 rejected;
- unsupported blinds: 300/300 rejected;
- fractional balance: 300/300 rejected;
- bad board-count/betround combination: 300/300 rejected.

Failures: **0**.

## Validated OpenHoldem interpretation

The gate uses the repository/OpenHoldem source semantics:

- `balanceN` = chair balance/stack behind;
- `currentbetN` = current amount of chips in play for that chair;
- `pot` = total amount of chips in play including player bets;
- `playersdealtbits` = chairs dealt this hand;
- `playersplayingbits` = chairs still playing;
- `playersallinbits` = all-in chairs;
- `betround` 1..4 = preflop..river;
- `ncommoncardsknown` = currently visible community-card count.

## Decision

The raw symbol/chair normalization layer is accepted.

The remaining gap is that the previous action reconciler consumed a richer
solver PublicSnapshot, while real OpenHoldem provides only the observable
projection.

The next end-to-end gate therefore runs:

`raw OpenHoldem frame -> strict adapter -> observable action reconciler -> canonical transcript -> from-scratch solver rebuild -> Hero SPNNIV1/SPNNIV2 + lean action parity`

This explicitly exercises real street reveals replacing earlier hidden filler
board cards.
