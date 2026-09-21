# SpinCore — LT2 runtime hidden-filler invariance result

Date: 2026-09-21  
Status: **PASS — HIDDEN OPPONENT/FUTURE BOARD FILLERS ARE SAFE FOR CURRENT-STATE RECONSTRUCTION**

## Scope

The audit changed only information unavailable to a real OpenHoldem runtime:

- opponent private hole cards;
- unrevealed future board cards.

It preserved:

- current Hero hole cards;
- visible board;
- public action path;
- tournament scenario.

No strategy inference, EV evaluation, optimizer work, training roots or holdout reuse occurred.

## Coverage

- target states: `6,000`;
- alternate hidden-card fillers checked: `36,000`;
- legal exact-action resolutions checked: `145,734`.

Domain coverage:

- THREE_HANDED: `4,066`;
- TRUE_HEADS_UP: `1,934`.

Street coverage:

- preflop: `2,007`;
- flop: `1,657`;
- turn: `1,306`;
- river: `1,030`.

## Result

Failures: **0**.

Across every alternate hidden-card completion, the following stayed exact:

- actor;
- domain;
- SPNNIV1 observation;
- SPNNIV2 public metadata;
- active action mask;
- lean legal actions;
- exact resolution of every legal lean action;
- visible-board count.

## Decision

A runtime reconstruction may use deterministic legal filler cards for:

- hidden opponent holes;
- unrevealed future board cards.

Those fillers must never overwrite Hero cards or already-visible board cards.

Because future real board cards can differ from an earlier filler, the production bridge should rebuild the canonical solver state from the hand start when new public cards appear, using the accumulated exact public action transcript.

The next gate therefore proves exact transcript-based from-scratch reconstruction at Hero decisions.
