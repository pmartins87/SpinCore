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
* AllinAdjustment::ChangeBetsizeToAllin converts f$betsize / technical bet-pot
  actions to all-in using f$allin_on_betsize_balance_ratio before casino entry.

The exact benchmark also mirrors the OpenHoldem SWAG amount-to adjustment that
precedes casino entry: RoundToBeautifulBetsize, legal min/max clamping and
tournament full-chip truncation. The benchmark uses raise-to semantics (the
ManualMode parity tablemap uses betsizeinterpretationmethod=3), so casino-specific
text interpretation is intentionally not applied to the simulator action.
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
    "betthirdpot": 0.333,
    "raisethirdpot": 0.333,
    "bethalfpot": 0.50,
    "raisehalfpot": 0.50,
    "bettwothirdpot": 0.667,
    "raisetwothirdpot": 0.667,
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


def _openholdem_round(value: float, unit: float = 1.0) -> float:
    """Mirror NumericalFunctions::Rounding for non-negative bet sizes."""
    value = float(value)
    unit = float(unit)
    if not math.isfinite(value) or not math.isfinite(unit):
        raise DeepCrusherActionTranslationError(
            f"non-finite OpenHoldem rounding input: value={value!r} unit={unit!r}"
        )
    if value < 0.0 or unit <= 0.0:
        raise DeepCrusherActionTranslationError(
            f"invalid OpenHoldem rounding input: value={value!r} unit={unit!r}"
        )
    return float(int((value + 0.5 * unit) / unit)) * unit


def _maximum_balance_raise_to(public, actor: int) -> float:
    try:
        current_bet = float(public.street_commitments[int(actor)])
        balance = float(public.stacks[int(actor)])
    except (AttributeError, IndexError, TypeError) as exc:
        raise DeepCrusherActionTranslationError(
            "OpenHoldem betsize adjustment requires stacks and street_commitments"
        ) from exc
    return current_bet + balance


def _round_to_beautiful_betsize(
    target_chips: float,
    *,
    public,
    actor: int,
    small_blind_chips: int,
    big_blind_chips: int,
) -> float:
    """Mirror SwagAdjustment::RoundToBeautifulBetsize."""
    value = float(target_chips)
    sb = float(small_blind_chips)
    bb = float(big_blind_chips)
    if not math.isfinite(value) or value < 0.0:
        raise DeepCrusherActionTranslationError(
            f"invalid raw betsize: {target_chips!r}"
        )
    if sb <= 0.0 or bb <= 0.0:
        raise DeepCrusherActionTranslationError(
            f"invalid blinds for betsize adjustment: sb={sb} bb={bb}"
        )

    maximum_balance = _maximum_balance_raise_to(public, actor)
    if maximum_balance > 0.0 and value >= maximum_balance:
        result = value
    elif value < 2.6 * bb:
        result = value
    elif value == _openholdem_round(value, bb):
        result = value
    elif value < 10.0 * sb:
        result = _openholdem_round(value, sb)
    else:
        result = _openholdem_round(value, bb)

    if result > 25000.0:
        result = _openholdem_round(result, 1000.0)
    elif result > 2500.0:
        result = _openholdem_round(result, 100.0)
    elif result > 250.0:
        result = _openholdem_round(result, 10.0)
    elif result > 20.0:
        result = _openholdem_round(result, 1.0)
    elif result > 0.90 and abs(result - int(result)) < 0.10:
        result = _openholdem_round(result, 1.0)

    return float(result)


def _openholdem_adjusted_raise_to(
    target_chips: float,
    *,
    public,
    actor: int,
    small_blind_chips: int,
    big_blind_chips: int,
    is_tournament: bool,
) -> int:
    """Return the final raise-to amount OpenHoldem would enter in ManualMode.

    This stops before SwagAmountAjustedToCasino because the exact simulator
    stores actual raise-to amounts. The canonical ManualMode tablemap uses
    betsizeinterpretationmethod=3, where the entered value is already raise-to.
    """
    low = int(public.min_raise_to)
    high = int(public.max_raise_to)
    if high <= 0 or low <= 0 or high < low:
        raise DeepCrusherActionTranslationError(
            f"invalid exact raise bounds: min={low} max={high}"
        )

    adjusted = _round_to_beautiful_betsize(
        target_chips,
        public=public,
        actor=actor,
        small_blind_chips=small_blind_chips,
        big_blind_chips=big_blind_chips,
    )
    adjusted = max(float(low), min(float(high), adjusted))

    if bool(is_tournament):
        maximum_balance = _maximum_balance_raise_to(public, actor)
        if maximum_balance <= 0.0:
            raise DeepCrusherActionTranslationError(
                "invalid maximum balance for tournament betsize adjustment"
            )
        if adjusted < maximum_balance:
            # RoundedBetsizeForTournaments truncates, then
            # AdaptValueToMinMaxRange can only move the value downward.
            adjusted = float(int(adjusted))

    # Solver exact actions are integer-chip raise-to values.
    target = _round_chip(adjusted)
    return max(low, min(high, target))


def _sized_aggression(
    public,
    actor: int,
    target_chips: float,
    *,
    small_blind_chips: int,
    big_blind_chips: int,
    is_tournament: bool,
) -> ExternalExactAction:
    kind = _aggressive_kind(public)
    if kind is None:
        return _call_with_backup(public)

    target = _openholdem_adjusted_raise_to(
        target_chips,
        public=public,
        actor=actor,
        small_blind_chips=small_blind_chips,
        big_blind_chips=big_blind_chips,
        is_tournament=is_tournament,
    )
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


def _change_betsize_to_allin(
    public,
    actor: int,
    target_chips: float,
    ratio: float | None,
) -> bool:
    """Mirror OpenHoldem AllinAdjustment::ChangeBetsizeToAllin.

    OpenHoldem compares the raw pre-casino betsize against
    f$allin_on_betsize_balance_ratio * (currentbet + remaining balance).
    It deliberately cancels the conversion when currentbet is already at or
    above the critical threshold, treating that state conservatively.
    """
    if ratio is None or float(ratio) <= 0.0:
        return False
    if not math.isfinite(float(ratio)):
        raise DeepCrusherActionTranslationError(
            f"non-finite allin_on_betsize_balance_ratio: {ratio!r}"
        )
    try:
        balance = float(public.stacks[int(actor)])
        current_bet = float(public.street_commitments[int(actor)])
    except (AttributeError, IndexError, TypeError) as exc:
        raise DeepCrusherActionTranslationError(
            "all-in adjustment requires public stacks and street_commitments"
        ) from exc
    if balance <= 0.0:
        return False
    total_balance = current_bet + balance
    if total_balance <= 0.0:
        return False
    critical_betsize = float(ratio) * total_balance
    if current_bet >= critical_betsize:
        return False
    return float(target_chips) >= critical_betsize


def _adjusted_sized_aggression(
    public,
    actor: int,
    target_chips: float,
    ratio: float | None,
    *,
    small_blind_chips: int,
    big_blind_chips: int,
    is_tournament: bool,
) -> ExternalExactAction:
    # AllinAdjustment happens before SWAG/beautiful-number adjustment.
    if _change_betsize_to_allin(public, actor, target_chips, ratio):
        return _allin_with_backup(public, actor)
    return _sized_aggression(
        public,
        actor,
        target_chips,
        small_blind_chips=small_blind_chips,
        big_blind_chips=big_blind_chips,
        is_tournament=is_tournament,
    )


def _fixed_action(
    name: str,
    public,
    actor: int,
    allin_adjustment_ratio: float | None = None,
    *,
    small_blind_chips: int,
    big_blind_chips: int,
    is_tournament: bool,
) -> ExternalExactAction:
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
        return _adjusted_sized_aggression(
            public,
            actor,
            target,
            allin_adjustment_ratio,
            small_blind_chips=small_blind_chips,
            big_blind_chips=big_blind_chips,
            is_tournament=is_tournament,
        )

    raise DeepCrusherActionTranslationError(f"unsupported fixed OpenPPL action: {name!r}")


def openppl_history_origin(
    decision: ReturnValue | DirectAction,
    action: ExternalExactAction,
) -> str:
    """Classify the *executed* OpenHoldem autoplayer action for history symbols.

    OpenHoldem distinguishes a minimum Raise button (prevaction=2 / didrais)
    from f$betsize and technical pot-size buttons (prevaction=3 / didbetsize).
    Poker-equivalent exact simulator actions alone cannot recover that
    distinction, so the oracle records it at translation time.
    """
    actual = int(action.action_type)
    if actual == ACTION_FOLD:
        return "fold"
    if actual == ACTION_CHECK:
        return "check"
    if actual == ACTION_CALL:
        return "call"
    if actual == ACTION_ALL_IN:
        return "allin"
    if actual not in (ACTION_BET_TO, ACTION_RAISE_TO):
        raise DeepCrusherActionTranslationError(
            f"unsupported executed action type for history: {actual}"
        )

    if isinstance(decision, DirectAction):
        low = str(decision.name).lower()
        if low in {"raise", "raisemin", "bet", "betmin"}:
            return "raise"
        # RaiseTo / RaiseBy and all technical pot-size actions are implemented
        # through f$betsize in OpenHoldem.
        return "betsize"

    if not isinstance(decision, ReturnValue):
        raise DeepCrusherActionTranslationError(
            f"unexpected OpenPPL decision type for history: {type(decision).__name__}"
        )
    value = float(decision.value)
    if value < -1000.0:
        rounded = int(round(value))
        if rounded == -1000012:
            return "raise"
        # RaiseMax would normally execute ALL_IN; if the simulator had to use an
        # aggressive backup, OpenPPL's first aggressive backup is a technical
        # pot-size/betsize action.
        return "betsize"
    # Positive and percent-pot numerical decisions are f$betsize.
    return "betsize"


def translate_openppl_decision(
    decision: ReturnValue | DirectAction,
    *,
    public,
    actor: int,
    big_blind_chips: int,
    small_blind_chips: int | None = None,
    is_tournament: bool = True,
    allin_adjustment_ratio: float | None = None,
) -> ExternalExactAction:
    """Translate one OpenPPL main-callback result to one exact simulator action."""
    bb = int(big_blind_chips)
    if bb <= 0:
        raise DeepCrusherActionTranslationError("big_blind_chips must be positive")
    sb = int(small_blind_chips) if small_blind_chips is not None else max(1, bb // 2)
    if sb <= 0:
        raise DeepCrusherActionTranslationError("small_blind_chips must be positive")

    if isinstance(decision, DirectAction):
        name = str(decision.name)
        low = name.lower()
        if low == "raiseto":
            if decision.amount is None or decision.amount_kind != "bb_expression":
                raise DeepCrusherActionTranslationError("RaiseTo missing BB expression")
            target = float(decision.amount) * bb
            return _adjusted_sized_aggression(
                public,
                actor,
                target,
                allin_adjustment_ratio,
                small_blind_chips=sb,
                big_blind_chips=bb,
                is_tournament=is_tournament,
            )
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
            return _adjusted_sized_aggression(
                public,
                actor,
                target,
                allin_adjustment_ratio,
                small_blind_chips=sb,
                big_blind_chips=bb,
                is_tournament=is_tournament,
            )
        return _fixed_action(
            name,
            public,
            actor,
            allin_adjustment_ratio,
            small_blind_chips=sb,
            big_blind_chips=bb,
            is_tournament=is_tournament,
        )

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
        target = value * bb
        return _adjusted_sized_aggression(
            public,
            actor,
            target,
            allin_adjustment_ratio,
        )

    if -1000.0 <= value < 0.0:
        # Small negative encoding: -0.50 means 50% pot.
        target = _pot_fraction_target(public, actor, -value)
        return _adjusted_sized_aggression(
            public,
            actor,
            target,
            allin_adjustment_ratio,
        )

    if value == 0.0:
        return _check_fold(public)

    rounded = int(round(value))
    if abs(value - rounded) > 1.0e-9 or rounded not in FIXED_ACTION_CODES:
        raise DeepCrusherActionTranslationError(
            f"unknown fixed OpenPPL action code: {value}"
        )
    return _fixed_action(
        FIXED_ACTION_CODES[rounded],
        public,
        actor,
        allin_adjustment_ratio,
        small_blind_chips=sb,
        big_blind_chips=bb,
        is_tournament=is_tournament,
    )
