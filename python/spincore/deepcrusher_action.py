from __future__ import annotations

"""Translate frozen DeepCrusher/OpenPPL decisions into simulator exact actions.

The rules mirror the OpenHoldem sources preserved in pmartins87/myoh_private:

* CAutoplayerFunctions::TranslateOpenPPLDecisionToAutoplayerFunctions
  - positive decision: final f$betsize in big blinds;
  - -1000 <= decision < 0: percent-pot f$betsize;
  - decision < -1000: fixed OpenPPL action constant;
  - zero: check/fold.
* CParseTreeTerminalNodeBetsizeAction
  - RaiseTo N: final wager N big blinds;
  - RaiseBy N: ncallbets + N big blinds;
  - RaiseBy X%: ncallbets + X% of pot-after-call.
* OpenPPL backup actions degrade unavailable aggression to call/check/fold.

Casino text-entry rounding is not part of poker strategy. The offline simulator has
integer chips, so fractional chip targets use deterministic half-up rounding and
are then clamped to the simulator's exact legal min/max interval. Real OpenHoldem
parity fixtures remain the final gate for this rounding convention.
"""

from decimal import Decimal, ROUND_HALF_UP
import math

from spincore.deepcrusher_benchmark import ExternalExactAction
from spincore.openppl_program import DirectAction, ReturnValue


ACTION_FOLD = 0
ACTION_CHECK = 1
ACTION_CALL = 2
ACTION_BET_TO = 3
ACTION_RAISE_TO = 4
ACTION_ALL_IN = 5

# Numeric values from the pinned OpenPPL library.
FIXED_ACTION_CODES = {
    -1000001: "Fold",
    -1000003: "RaiseFourthPot",
    -1000004: "RaiseThirdPot",
    -1000005: "RaiseHalfPot",
    -1000006: "RaiseTwoThirdPot",
    -1000007: "RaiseThreeFourthPot",
    -1000008: "RaisePot",
    -1000009: "RaiseMax",
    -1000010: "Call",
    -1000012: "RaiseMin",
}

_POT_FRACTION_BY_NAME = {
    "betfourthpot": 0.25,
    "raisefourthpot": 0.25,
    "betthirdpot": 1.0 / 3.0,
    "raisethirdpot": 1.0 / 3.0,
    "bethalfpot": 0.50,
    "raisehalfpot": 0.50,
    "bettwothirdpot": 2.0 / 3.0,
    "raisetwothirdpot": 2.0 / 3.0,
    "betthreefourthpot": 0.75,
    "raisethreefourthpot": 0.75,
    "betpot": 1.0,
    "raisepot": 1.0,
}

class DeepCrusherActionTranslationError(RuntimeError):
    pass


def _round_chip(value: float) -> int:
    if not math.isfinite(float(value)):
        raise DeepCrusherActionTranslationError(f"non-finite chip target: {value!r}")
    return int(Decimal(str(float(value))).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _fallback_passive(public) -> ExternalExactAction:
    if bool(public.legal_call):
        return ExternalExactAction(ACTION_CALL)
    if bool(public.legal_check):
        return ExternalExactAction(ACTION_CHECK)
    if bool(public.legal_fold):
        return ExternalExactAction(ACTION_FOLD)
    if bool(public.legal_all_in):
        # Rare fold/all-in-only UI-equivalent state.
        return ExternalExactAction(ACTION_ALL_IN)
    raise DeepCrusherActionTranslationError("state exposes no executable passive action")


def _check_fold(public) -> ExternalExactAction:
    if bool(public.legal_check):
        return ExternalExactAction(ACTION_CHECK)
    if bool(public.legal_fold):
        return ExternalExactAction(ACTION_FOLD)
    return _fallback_passive(public)


def _call_with_backup(public) -> ExternalExactAction:
    if bool(public.legal_call):
        return ExternalExactAction(ACTION_CALL)
    if bool(public.legal_check):
        return ExternalExactAction(ACTION_CHECK)
    if bool(public.legal_all_in) and not bool(public.legal_fold):
        return ExternalExactAction(ACTION_ALL_IN)
    if bool(public.legal_fold):
        return ExternalExactAction(ACTION_FOLD)
    raise DeepCrusherActionTranslationError("Call has no legal backup action")


def _aggressive_kind(public) -> int | None:
    if bool(public.legal_raise):
        return ACTION_RAISE_TO
    if bool(public.legal_bet):
        return ACTION_BET_TO
    return None


def _sized_aggression(public, target_chips: float) -> ExternalExactAction:
    kind = _aggressive_kind(public)
    if kind is None:
        return _call_with_backup(public)

    low = int(public.min_raise_to)
    high = int(public.max_raise_to)
    if high <= 0 or low <= 0 or high < low:
        raise DeepCrusherActionTranslationError(
            f"invalid exact raise bounds: min={low} max={high}"
        )
    target = max(low, min(high, _round_chip(target_chips)))
    return ExternalExactAction(kind, target)


def _pot_fraction_target(public, actor: int, fraction: float) -> float:
    if fraction < 0:
        raise DeepCrusherActionTranslationError("negative pot fraction")
    hero_bet = int(public.street_commitments[int(actor)])
    to_call = int(public.to_call)
    # OpenHoldem BetSizeForPercentagedPotsizeBet:
    # current hero bet + call + fraction * (pot + call).
    return hero_bet + to_call + float(fraction) * (int(public.pot) + to_call)


def _allin_with_backup(public, actor: int) -> ExternalExactAction:
    if bool(public.legal_all_in):
        return ExternalExactAction(ACTION_ALL_IN)
    kind = _aggressive_kind(public)
    if kind is not None:
        return ExternalExactAction(kind, int(public.max_raise_to))
    return _call_with_backup(public)


def _fixed_action(name: str, public, actor: int) -> ExternalExactAction:
    low = str(name).lower()

    if low in {"allin", "betmax", "raisemax"}:
        return _allin_with_backup(public, actor)
    if low in {"betmin", "raisemin", "bet", "raise"}:
        kind = _aggressive_kind(public)
        if kind is None:
            return _call_with_backup(public)
        return ExternalExactAction(kind, int(public.min_raise_to))
    if low == "call":
        return _call_with_backup(public)
    if low == "check":
        return _check_fold(public)
    if low == "fold":
        # In OpenPPL, zero/empty is check-fold. A literal fold can also land in
        # a free-action state on sites without a fold button, where check is the
        # only poker-equivalent legal backup.
        if bool(public.legal_fold):
            return ExternalExactAction(ACTION_FOLD)
        if bool(public.legal_check):
            return ExternalExactAction(ACTION_CHECK)
        return _fallback_passive(public)
    if low in _POT_FRACTION_BY_NAME:
        target = _pot_fraction_target(public, actor, _POT_FRACTION_BY_NAME[low])
        return _sized_aggression(public, target)

    raise DeepCrusherActionTranslationError(f"unsupported fixed OpenPPL action: {name!r}")


def translate_openppl_decision(
    decision: ReturnValue | DirectAction,
    *,
    public,
    actor: int,
    big_blind_chips: int,
) -> ExternalExactAction:
    """Translate one OpenPPL main-callback result to one exact simulator action."""
    bb = int(big_blind_chips)
    if bb <= 0:
        raise DeepCrusherActionTranslationError("big_blind_chips must be positive")

    if isinstance(decision, DirectAction):
        name = str(decision.name)
        low = name.lower()
        if low == "raiseto":
            if decision.amount is None or decision.amount_kind != "bb_expression":
                raise DeepCrusherActionTranslationError("RaiseTo missing BB expression")
            return _sized_aggression(public, float(decision.amount) * bb)
        if low == "raiseby":
            if decision.amount is None:
                raise DeepCrusherActionTranslationError("RaiseBy missing amount")
            if decision.amount_kind == "pot_fraction":
                target = _pot_fraction_target(public, actor, float(decision.amount))
            elif decision.amount_kind == "bb_expression":
                ncall_target = (
                    int(public.street_commitments[int(actor)]) + int(public.to_call)
                )
                target = ncall_target + float(decision.amount) * bb
            else:
                raise DeepCrusherActionTranslationError(
                    f"unsupported RaiseBy amount kind: {decision.amount_kind!r}"
                )
            return _sized_aggression(public, target)
        return _fixed_action(name, public, actor)

    if not isinstance(decision, ReturnValue):
        raise DeepCrusherActionTranslationError(
            f"unexpected OpenPPL decision type: {type(decision).__name__}"
        )

    value = float(decision.value)
    if not math.isfinite(value):
        raise DeepCrusherActionTranslationError(f"non-finite OpenPPL decision: {value}")

    if value > 0:
        # TranslateOpenPPLDecisionToAutoplayerFunctions: positive values are
        # final f$betsize values in big blinds.
        return _sized_aggression(public, value * bb)

    if -1000.0 <= value < 0.0:
        # Small negative encoding: -0.50 means 50% pot.
        target = _pot_fraction_target(public, actor, -value)
        return _sized_aggression(public, target)

    if value == 0.0:
        return _check_fold(public)

    rounded = int(round(value))
    if abs(value - rounded) > 1.0e-9 or rounded not in FIXED_ACTION_CODES:
        raise DeepCrusherActionTranslationError(
            f"unknown fixed OpenPPL action code: {value}"
        )
    return _fixed_action(FIXED_ACTION_CODES[rounded], public, actor)
