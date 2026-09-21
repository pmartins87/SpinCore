from __future__ import annotations

"""Deterministic one-action public-snapshot reconciler for runtime tracking.

The input state is authoritative SpinCore state before one observed voluntary
action. The observed snapshot is table-public state after that action.

The reconciler deliberately canonicalizes economically equivalent all-in
aliases to the action convention used by the validated LT2 lean strategy:
- an all-in that merely calls -> Call;
- an aggressive action that empties the stack -> AllIn;
- otherwise use BetTo or RaiseTo according to the pre-action current bet.

The inferred action is accepted only if applying it through the authoritative
solver reproduces the observed public snapshot exactly.
"""

from dataclasses import dataclass

from spincore.solver import PublicSnapshot, ResolvedExactAction


class RuntimeReconciliationError(RuntimeError):
    pass


def infer_single_exact_action(state, observed: PublicSnapshot) -> ResolvedExactAction:
    before = state.public_snapshot()
    if before.terminal:
        raise RuntimeReconciliationError("cannot reconcile from terminal state")
    actor = int(before.actor)
    if actor not in (0, 1, 2):
        raise RuntimeReconciliationError("invalid pre-action actor")

    if observed.domain != before.domain:
        raise RuntimeReconciliationError("domain changed across one public action")
    if observed.stacks[actor] > before.stacks[actor]:
        raise RuntimeReconciliationError("acting stack increased")
    if before.folded[actor] or before.all_in[actor]:
        raise RuntimeReconciliationError("pre-action actor is not actionable")

    paid = int(before.stacks[actor] - observed.stacks[actor])
    folded_now = bool(observed.folded[actor]) and not bool(before.folded[actor])

    if folded_now:
        if paid != 0:
            raise RuntimeReconciliationError("fold transition paid chips")
        candidate = ResolvedExactAction(0, 0)  # Fold
    elif paid == 0:
        candidate = ResolvedExactAction(1, 0)  # Check
    else:
        target = int(before.street_commitments[actor] + paid)
        # Canonical LT2 convention: a short all-in that merely matches as much
        # of the outstanding wager as possible is represented by Call.
        if before.to_call > 0 and target <= before.current_bet:
            candidate = ResolvedExactAction(2, 0)  # Call
        # Any aggressive action that consumes the actor's remaining stack is
        # represented by AllIn, not BetTo/RaiseTo-to-max.
        elif int(observed.stacks[actor]) == 0:
            candidate = ResolvedExactAction(5, 0)  # AllIn
        elif before.current_bet == 0:
            candidate = ResolvedExactAction(3, target)  # BetTo
        else:
            candidate = ResolvedExactAction(4, target)  # RaiseTo

    probe = state.clone()
    try:
        try:
            probe.apply_exact(candidate.action_type, candidate.amount_to)
        except Exception as exc:
            raise RuntimeReconciliationError(
                f"inferred exact action is illegal: {candidate}"
            ) from exc
        got = probe.public_snapshot()
    finally:
        probe.close()

    if got != observed:
        raise RuntimeReconciliationError(
            "one-action reconciliation does not reproduce observed public snapshot"
        )
    return candidate
