# SpinCore Current Work

Date: 2026-09-21
Status: **OPENHOLDEM OBSERVABLE E2E — ALL-IN RUNOUT BETROUND SEMANTICS PATCHED**

## Frozen strategy/runtime

All strategic/model identities remain frozen.

No training, EV tuning or holdout reuse is permitted.

## Latest E2E failure

The rerun did not reach a strategic comparison. It stopped inside the strict
OpenHoldem symbol adapter with:

`board-count/betround mismatch: street=1 visible=5`

This occurred after an all-in runout.

## Root cause

OpenHoldem's `CBetroundCalculator` derives the betting round from which
community cards are actually known:

- river card known -> river;
- else turn card known -> turn;
- else first three common cards known -> flop;
- else preflop.

SpinCore's terminal hand engine has different internal semantics after an all-in
runout: when only one player remains actionable, it reveals all five board cards
and terminates without advancing the internal betting street through every
remaining round.

Therefore a terminal state can legitimately be:

- SpinCore internal betting street = flop/turn/preflop;
- visible board count = 5;
- OpenHoldem betround = river.

The previous synthetic OH frame incorrectly set `betround = solver.street+1`,
creating an impossible OpenHoldem frame such as flop + five visible board cards.

## Patch

The runtime integration now mirrors OpenHoldem's real card-derived semantics:

1. `openholdem_betround_from_visible_count()` maps visible board count
   0/3/4/5 -> OH betround 1/2/3/4;
2. synthetic E2E and symbol-adapter frames use that mapping;
3. canonical observable projection also uses visible board count rather than
   SpinCore's internal last betting street;
4. observable reconciliation permits a forward OH betround jump caused by a
   single all-in runout, but still requires authoritative `apply_exact` plus
   full-frame equality before accepting it.

No strategy, model, action sizing or solver betting rule was changed.

## Previous invisible-CHECK patch remains active

OpenHoldem snapshots cannot directly reveal a CHECK. Pending silent checks are
still synchronized using later observable evidence or `DLLUpdateOnMyTurn`.

## Immediate action

```bash
bash tools/run_lt2_openholdem_observable_tracker_e2e.sh
```

Wait for `LT2_OPENHOLDEM_OBSERVABLE_TRACKER_E2E_PASS`.

If it fails again, do not rerun. Send the terminal output and, if created, the
latest `SpinCore_LT2_openholdem_observable_tracker_e2e.json`.
