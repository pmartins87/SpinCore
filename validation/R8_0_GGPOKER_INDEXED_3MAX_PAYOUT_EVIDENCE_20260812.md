# R8.0 — GGPoker indexed 3-Max payout/frequency evidence

Status: **PARTIAL FIRST-PARTY EVIDENCE / NOT SUFFICIENT FOR R8.0 PASS**

`READY FOR TABLES = NO`

Retrieval date: 2026-08-12/13 UTC boundary.

First-party source:

```text
https://legal.ggpoker.com/es/poker-games/spin-gold/
```

## What the current first-party search index exposes

The current GGPoker legal Spin & Gold page exposes the 3-Max buy-in selector set:

```text
$0.25, $1, $3, $5, $10, $20, $50, $100, $200
```

A current search-index rendering of that same first-party page also exposes this complete 3-Max payout/frequency row set:

```text
multiplier   1st       2nd       3rd       frequency / 100,000,000
x40,000      x20,000   x12,000   x8,000    50
x100         x60       x30       x10       4,000
x50          x30       x15       x5        8,000
x10          x8        x2        0         2,350,000
x5           x5        0         0         2,500,000
x4           x4        0         0         2,650,000
x3           x3        0         0         44,624,100
x2           x2        0         0         47,863,850
```

The frequencies sum exactly to 100,000,000.

Normalized payout-share semantics implied by the displayed row arithmetic, conditional on this exact row set being bound to a production state, are:

```text
x40,000 -> [0.50, 0.30, 0.20]
x100    -> [0.60, 0.30, 0.10]
x50     -> [0.60, 0.30, 0.10]
x10     -> [0.80, 0.20, 0.00]
x5      -> [1.00, 0.00, 0.00]
x4      -> [1.00, 0.00, 0.00]
x3      -> [1.00, 0.00, 0.00]
x2      -> [1.00, 0.00, 0.00]
```

These normalized shares are arithmetic transformations of the displayed first-party prize multipliers; they are not an independent rule claim.

## Critical selector-binding limitation

This evidence is **not accepted as a production profile table** because the search-index representation does not prove which 3-Max buy-in button is selected when the rows are rendered.

That ambiguity is material. The same first-party page explicitly states that **$5 buy-in tables can offer up to x200,000**, whereas the indexed row set above tops out at x40,000. Therefore the row set cannot be silently generalized to every buy-in and cannot be assigned to `$5` merely because `$5` is one of the visible selector buttons.

The raw/open page representation visible to the current retrieval path shows the selector buttons and table headers but does not preserve the dynamic row-to-selected-button binding. Consequently any guess that the first/default button `$0.25` owns the exposed rows remains an inference and is **not** sufficient for `SELECTED_PROFILE_STATE` evidence.

## Structure gap remains

The same current first-party page states that starting chips, blind structure and time bank differ according to the Spin Multiplier. It does not expose the complete multiplier-bound stack/blind/time-bank table through the current retrieval representation.

A current first-party GGPoker strategy article states descriptively that ordinary Spin & Gold starts at 500 chips, increases for larger multipliers and uses hyper-turbo blinds increasing every 3 minutes. This is useful corroboration of multiplier dependence, but it is not a complete enumerated selected-state structure table and must not fill missing high-multiplier values.

## R8.0 consequence

This new evidence narrows the unresolved R8.0 work but does not close it.

Accepted use now:

- preserve the exact selector-unbound 3-Max payout/frequency row set as first-party evidence;
- use its arithmetic payout shares only after an exact selected-state binding proves applicability;
- recognize that payout topology changes with multiplier.

Still required before any production profile PASS:

1. exact binding of payout/frequency rows to the selected `table_size × buy_in × multiplier` state;
2. exact starting chips for each accepted multiplier state;
3. exact full blind/ante sequence for each accepted state;
4. exact time-bank structure where operationally relevant;
5. proof of the multiplier set applicable to each accepted buy-in, including the special `$5` maximum advertised by GGPoker;
6. target jurisdiction/account/client binding where rules can differ.

No R7.4 pilot constants may substitute for these missing values.

`R8.0 production profile gate = NOT PASS`.
