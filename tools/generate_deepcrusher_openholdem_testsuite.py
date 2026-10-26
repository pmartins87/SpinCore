#!/usr/bin/env python3
from __future__ import annotations

"""Generate deterministic OpenHoldem TestSuite2 parity cases for DeepCrusher R8.

The expected hero actions come from the offline R8 oracle. The generated files
are then intended to be executed by the *real* OpenHoldem ManualMode/TestSuite2
with the frozen R8 formula loaded. Equality of action button and exact NL
betsize is the DC0 parity gate.

This generator uses only passive opponents (check/call, otherwise fold) so the
same hand can usually exercise several DeepCrusher decisions across streets.
Cases that terminate before the requested street coverage are discarded.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from spincore.deepcrusher_benchmark import ExternalExactAction  # noqa: E402
from spincore.deepcrusher_policy import DeepCrusherR8Policy  # noqa: E402
from spincore.solver import Episode, SolverLibrary  # noqa: E402


ACTION_TEXT = {
    0: "F",
    1: "K",
    2: "C",
    3: "R",
    4: "R",
    5: "A",
}
RANKS = "23456789TJQKA"
SUITS = "shdc"


@dataclass
class RenderedAction:
    street: int
    actor: int
    hero: bool
    action_type: int
    amount_to: int
    legal_buttons: str = ""
    decision_family: str = ""
    decision_metadata: dict[str, object] | None = None


_FIXED_RETURN_FAMILY = {
    -1000001: "FOLD",
    -1000003: "TECH_BETPOT_025",
    -1000004: "TECH_BETPOT_033",
    -1000005: "TECH_BETPOT_050",
    -1000006: "TECH_BETPOT_067",
    -1000007: "TECH_BETPOT_075",
    -1000008: "TECH_BETPOT_100",
    -1000009: "DIRECT_ALLIN",
    -1000010: "CALL",
    -1000012: "MIN_RAISE",
}


def classify_decision_family(
    metadata: dict[str, object] | None,
    action: ExternalExactAction,
) -> str:
    """Describe the OpenPPL/autoplayer path that produced the executed action.

    The label is evidence metadata only. It does not participate in policy
    selection or testcase rendering.
    """
    if metadata is None:
        return "UNKNOWN"

    kind = str(metadata.get("openppl_result_kind", ""))
    actual = int(action.action_type)

    if kind == "DirectAction":
        name = str(metadata.get("openppl_action_name", "")).lower()
        if actual == 5 and name not in {"allin", "betmax", "raisemax"}:
            return "ALLIN_ADJUSTED"
        if name in {"allin", "betmax", "raisemax"}:
            return "DIRECT_ALLIN"
        if name == "raiseto":
            return "RAISETO_BB"
        if name == "raiseby":
            amount_kind = str(metadata.get("openppl_action_amount_kind", ""))
            return (
                "RAISEBY_POT"
                if amount_kind == "pot_fraction"
                else "RAISEBY_BB"
            )
        named = {
            "betfourthpot": "TECH_BETPOT_025",
            "raisefourthpot": "TECH_BETPOT_025",
            "betthirdpot": "TECH_BETPOT_033",
            "raisethirdpot": "TECH_BETPOT_033",
            "bethalfpot": "TECH_BETPOT_050",
            "raisehalfpot": "TECH_BETPOT_050",
            "bettwothirdpot": "TECH_BETPOT_067",
            "raisetwothirdpot": "TECH_BETPOT_067",
            "betthreefourthpot": "TECH_BETPOT_075",
            "raisethreefourthpot": "TECH_BETPOT_075",
            "betpot": "TECH_BETPOT_100",
            "raisepot": "TECH_BETPOT_100",
            "betmin": "MIN_RAISE",
            "raisemin": "MIN_RAISE",
            "bet": "MIN_RAISE",
            "raise": "MIN_RAISE",
            "fold": "FOLD",
            "call": "CALL",
            "check": "CHECK",
        }
        return named.get(name, "DIRECT_" + name.upper())

    if kind == "ReturnValue":
        value = float(metadata.get("openppl_return_value", 0.0))
        if actual == 5 and abs(value - (-1000009.0)) > 1.0e-9:
            return "ALLIN_ADJUSTED"
        if value > 0.0:
            return "F_BETSIZE_BB"
        if -1000.0 <= value < 0.0:
            return "F_BETSIZE_PERCENT"
        rounded = int(round(value))
        if abs(value - rounded) <= 1.0e-9 and rounded in _FIXED_RETURN_FAMILY:
            return _FIXED_RETURN_FAMILY[rounded]
        if value == 0.0:
            return "CHECK_FOLD"
        return "RETURN_OTHER"

    return "UNKNOWN"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cases-per-domain", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--max-search", type=int, default=400)
    p.add_argument(
        "--require-semantic-sentinels",
        action="store_true",
        help=(
            "continue deterministic search until adjusted-allin and literal "
            "33.3%/66.7% technical bet-pot decision families are captured"
        ),
    )
    return p.parse_args()


def card_text(card_id: int) -> str:
    value = int(card_id)
    if value < 0 or value >= 52:
        raise ValueError(value)
    return RANKS[value // 4] + SUITS[value % 4]


def card_id(text: str) -> int:
    raw = str(text)
    if len(raw) != 2 or raw[0] not in RANKS or raw[1] not in SUITS:
        raise ValueError(f"invalid card text: {text!r}")
    return RANKS.index(raw[0]) * 4 + SUITS.index(raw[1])


def explicit_husb_067_candidates() -> list[tuple[tuple[tuple[int, int], ...], tuple[int, ...]]]:
    """Small directed HUSB river-value probes for the 66.7% technical path.

    Hero is seat 0 / dealer-SB.  Every board is unpaired and deliberately
    non-four-flush/non-four-straight so a surviving top pair should reach the
    HUSB river-value branch that sets user_River66 before f$BestBetsize.
    """
    raw = [
        (("As", "Kc"), ("Qh", "Jd"), ("Ks", "7h", "2d", "4c", "9s")),
        (("Ah", "Qc"), ("Kh", "Jd"), ("Qs", "8h", "3d", "5c", "Ts")),
        (("Ks", "Qc"), ("Ah", "Jd"), ("Kd", "8s", "3h", "5c", "9d")),
        (("Qs", "Jc"), ("Ah", "Td"), ("Qd", "8s", "3h", "5c", "9d")),
        (("As", "Jc"), ("Kh", "Qd"), ("Js", "8h", "3d", "5c", "9s")),
        (("Kc", "Qh"), ("As", "Jd"), ("Kh", "8s", "2d", "4c", "Ts")),
    ]
    out = []
    for hero, villain, board in raw:
        holes = (
            tuple(card_id(x) for x in hero),
            tuple(card_id(x) for x in villain),
            (-1, -1),
        )
        board_ids = tuple(card_id(x) for x in board)
        out.append((holes, board_ids))
    return out


def explicit_hubb_067_candidates() -> list[tuple[tuple[tuple[int, int], ...], tuple[int, ...]]]:
    """Directed HU/BB limp-check-check probes for user_TurnProbeOpportunity."""
    raw = [
        (("As", "Kh"), ("7c", "2d"), ("Qs", "8h", "3d", "5c", "9s")),
        (("Ah", "Kd"), ("7c", "2s"), ("Ks", "8h", "3d", "5c", "9s")),
        (("As", "Qh"), ("6c", "2h"), ("Js", "8d", "3h", "5c", "9s")),
        (("Ah", "Qd"), ("Jc", "2h"), ("Ks", "8d", "3h", "5c", "9s")),
        (("As", "Kd"), ("Tc", "4h"), ("Qs", "8d", "3h", "6c", "9s")),
        (("Ah", "Qd"), ("9c", "2h"), ("Ks", "Qs", "3h", "5c", "7d")),
        (("As", "Jh"), ("8c", "2d"), ("Qs", "7h", "3d", "5c", "Ts")),
        (("Ah", "Jd"), ("7s", "3c"), ("Ks", "9h", "4d", "2c", "Qd")),
    ]
    out = []
    for villain, hero, board in raw:
        holes = (
            tuple(card_id(x) for x in villain),
            tuple(card_id(x) for x in hero),
            (-1, -1),
        )
        board_ids = tuple(card_id(x) for x in board)
        out.append((holes, board_ids))
    return out


def explicit_hubb_villain_init_067_candidates() -> list[tuple[tuple[tuple[int, int], ...], tuple[int, ...]]]:
    """HU/BB probes: villain min-raises preflop then checks back the flop."""
    raw = [
        (("As", "Kh"), ("Jc", "8c"), ("Qs", "7h", "3d", "5c", "2s")),
        (("Ah", "Kd"), ("Tc", "8c"), ("Qs", "7h", "3d", "5c", "2s")),
        (("As", "Qh"), ("9c", "8c"), ("Ks", "Qd", "3h", "5c", "2s")),
        (("Ah", "Qd"), ("Jc", "9c"), ("Ks", "8d", "3h", "5c", "2s")),
        (("As", "Jh"), ("Qc", "9c"), ("Ks", "8d", "3h", "5c", "2s")),
        (("Ah", "Jd"), ("Tc", "9c"), ("Ks", "7d", "3h", "5c", "2s")),
        (("As", "Kh"), ("Jd", "Td"), ("Qs", "8h", "3c", "5d", "2s")),
        (("Ah", "Kd"), ("9d", "7d"), ("Ks", "Qh", "3c", "5d", "2s")),
    ]
    out = []
    for villain, hero, board in raw:
        holes = (
            tuple(card_id(x) for x in villain),
            tuple(card_id(x) for x in hero),
            (-1, -1),
        )
        board_ids = tuple(card_id(x) for x in board)
        out.append((holes, board_ids))
    return out


def buttons(public) -> str:
    out = []
    if public.legal_fold:
        out.append("F")
    if public.legal_call:
        out.append("C")
    if public.legal_check:
        out.append("K")
    if public.legal_raise or public.legal_bet:
        out.append("R")
    if public.legal_all_in:
        out.append("A")
    return "".join(out)


def passive_action(public) -> ExternalExactAction:
    if public.legal_call:
        return ExternalExactAction(2)
    if public.legal_check:
        return ExternalExactAction(1)
    if public.legal_fold:
        return ExternalExactAction(0)
    if public.legal_all_in:
        return ExternalExactAction(5)
    raise RuntimeError("passive opponent has no legal action")


def min_aggressive_action(public) -> ExternalExactAction:
    if public.legal_raise:
        return ExternalExactAction(4, int(public.min_raise_to))
    if public.legal_bet:
        return ExternalExactAction(3, int(public.min_raise_to))
    return passive_action(public)


def allin_pressure_action(public) -> ExternalExactAction:
    if public.legal_all_in:
        return ExternalExactAction(5)
    return min_aggressive_action(public)


def opponent_action(public, style: str) -> ExternalExactAction:
    if style == "PASSIVE":
        return passive_action(public)
    if style == "MIN_AGGRO":
        return min_aggressive_action(public)
    if style == "PREFLOP_MIN_THEN_PASSIVE":
        return (
            min_aggressive_action(public)
            if int(public.street) == 0
            else passive_action(public)
        )
    if style == "ALLIN_PRESSURE":
        return allin_pressure_action(public)
    raise ValueError(f"unknown opponent style: {style}")


def blind_seats(episode: Episode) -> tuple[int, int]:
    live = [seat for seat, stack in enumerate(episode.stacks) if int(stack) > 0]
    dealer = int(episode.dealer_id)
    if episode.game_is_hu:
        if dealer not in live or len(live) != 2:
            raise ValueError("bad HU dealer/live seats")
        other = next(seat for seat in live if seat != dealer)
        return dealer, other
    if len(live) != 3:
        raise ValueError("3H episode requires three live seats")
    return (dealer + 1) % 3, (dealer + 2) % 3


def expected_token(action: RenderedAction) -> str:
    letter = ACTION_TEXT[int(action.action_type)]
    if letter == "R":
        return f"R {int(action.amount_to)}"
    return letter


def applied_token(name: str, action: RenderedAction) -> str:
    letter = ACTION_TEXT[int(action.action_type)]
    if letter == "R":
        return f"{name} R {int(action.amount_to)}"
    return f"{name} {letter}"


def hero_token(name: str, action: RenderedAction) -> str:
    return (
        f"{name} can {action.legal_buttons} do "
        f"{expected_token(action)}"
    )


def render_case(
    *,
    case_id: str,
    episode: Episode,
    deal,
    actions: list[RenderedAction],
    hero_seat: int,
) -> str:
    names = {0: "P0", 1: "P1", 2: "P2"}
    names[hero_seat] = "Bot"
    live = [seat for seat, stack in enumerate(episode.stacks) if int(stack) > 0]
    sb, bb = blind_seats(episode)

    pre = [f"{names[sb]} S", f"{names[bb]} B"]
    street_rows: dict[int, list[str]] = {0: pre, 1: [], 2: [], 3: []}
    for action in actions:
        token = (
            hero_token(names[action.actor], action)
            if action.hero
            else applied_token(names[action.actor], action)
        )
        street_rows[int(action.street)].append(token)

    if not any(a.hero for a in actions):
        raise ValueError("case has no hero action")

    balances = ",".join(
        f"{names[seat]} {int(episode.stacks[seat])}"
        for seat in live
    )
    hero_cards = ", ".join(card_text(x) for x in deal.holes[hero_seat])
    lines = [
        f"; generated parity case {case_id}",
        "[table]",
        f"sblind = {int(episode.small_blind)}",
        f"bblind = {int(episode.big_blind)}",
        "gtype = NL",
        "network = ggpoker",
        "tournament = true",
        f"balances = {balances}",
        "",
        "[preflop]",
        f"Hand = {hero_cards}",
        "Actions = " + ",".join(street_rows[0]),
    ]

    if street_rows[1]:
        lines += [
            "",
            "[flop]",
            "Cards = " + ", ".join(card_text(x) for x in deal.board[:3]),
            "Actions = " + ",".join(street_rows[1]),
        ]
    if street_rows[2]:
        lines += [
            "",
            "[turn]",
            "Card = " + card_text(deal.board[3]),
            "Actions = " + ",".join(street_rows[2]),
        ]
    if street_rows[3]:
        lines += [
            "",
            "[river]",
            "Card = " + card_text(deal.board[4]),
            "Actions = " + ",".join(street_rows[3]),
        ]
    return "\n".join(lines) + "\n"


def play_candidate(
    solver: SolverLibrary,
    policy: DeepCrusherR8Policy,
    episode: Episode,
    *,
    hero_seat: int,
    deal_seed: int,
    case_index: int,
    opponent_style: str = "PASSIVE",
    explicit_holes: tuple[tuple[int, int], ...] | None = None,
    explicit_board: tuple[int, ...] | None = None,
):
    lineup = ["PASSIVE", "PASSIVE", "DEAD" if episode.game_is_hu else "PASSIVE"]
    lineup[hero_seat] = policy.policy_id
    lineup_obj = type("LineupFixture", (), {"seats": tuple(lineup)})()
    policy.begin_hand(
        episode=episode,
        lineup=lineup_obj,
        scenario_index=case_index,
        lineup_index=0,
        deal_seed=deal_seed,
    )

    if (explicit_holes is None) != (explicit_board is None):
        raise ValueError("explicit_holes and explicit_board must be provided together")
    state = (
        solver.create(episode, deal_seed)
        if explicit_holes is None
        else solver.create_with_deal(episode, explicit_holes, explicit_board)
    )
    rendered: list[RenderedAction] = []
    deal = state.deal_snapshot()
    hero_streets: set[int] = set()
    try:
        steps = 0
        while not state.terminal:
            steps += 1
            if steps > 200:
                raise RuntimeError("candidate exceeded action limit")
            actor = int(state.actor)
            public = state.public_snapshot()
            if actor == hero_seat:
                action = policy.choose_exact(
                    state,
                    seat=actor,
                    rng=random.Random(deal_seed ^ steps),
                )
                metadata = policy.decision_metadata(seat=actor)
                hero_streets.add(int(public.street))
                rendered.append(
                    RenderedAction(
                        street=int(public.street),
                        actor=actor,
                        hero=True,
                        action_type=int(action.action_type),
                        amount_to=int(action.amount_to),
                        legal_buttons=buttons(public),
                        decision_family=classify_decision_family(metadata, action),
                        decision_metadata=metadata,
                    )
                )
            else:
                action = opponent_action(public, opponent_style)
                rendered.append(
                    RenderedAction(
                        street=int(public.street),
                        actor=actor,
                        hero=False,
                        action_type=int(action.action_type),
                        amount_to=int(action.amount_to),
                    )
                )
            state.apply_exact(int(action.action_type), int(action.amount_to))
        return deal, rendered, hero_streets
    finally:
        state.close()


def make_episode(
    domain: str,
    dealer: int,
    *,
    stack_chips: int | None = None,
) -> Episode:
    if domain == "THREE_HANDED":
        stack = 500 if stack_chips is None else int(stack_chips)
        return Episode(
            total_chips=3 * stack,
            game_is_hu=False,
            blind_index=0,
            small_blind=10,
            big_blind=20,
            stacks=(stack, stack, stack),
            dealer_id=dealer,
        )
    # Keep the dead seat fixed at 2 for simpler TestSuite2 rendering.
    stack = 750 if stack_chips is None else int(stack_chips)
    return Episode(
        total_chips=2 * stack,
        game_is_hu=True,
        blind_index=0,
        small_blind=10,
        big_blind=20,
        stacks=(stack, stack, 0),
        dealer_id=dealer,
        dead_players=(2,),
    )


def main() -> int:
    args = parse_args()
    solver = SolverLibrary(args.solver)
    if not solver.explicit_deal_available:
        raise SystemExit("parity generator requires explicit deal snapshot ABI")
    policy = DeepCrusherR8Policy.from_repository(ROOT)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema": "SPINCORE_DEEPCRUSHER_OPENHOLDEM_TESTSUITE2_PARITY_PACK_V1",
        "status": "GENERATED_EXPECTATIONS_FROM_OFFLINE_ORACLE",
        "canonical_parity_pass": False,
        "required_next_system": "REAL_OPENHOLDEM_MANUALMODE_TESTSUITE2",
        "cases": [],
    }

    required_actions = {"F", "C", "K", "R", "A"}
    covered_actions: set[str] = set()
    covered_decision_families: set[str] = set()
    required_decision_families = (
        {"ALLIN_ADJUSTED", "TECH_BETPOT_033", "TECH_BETPOT_067"}
        if args.require_semantic_sentinels
        else set()
    )
    covered_streets = {
        "THREE_HANDED": set(),
        "TRUE_HEADS_UP": set(),
    }
    full_street_counts = {
        "THREE_HANDED": 0,
        "TRUE_HEADS_UP": 0,
    }
    case_serial = {
        "THREE_HANDED": 0,
        "TRUE_HEADS_UP": 0,
    }

    for domain in ("THREE_HANDED", "TRUE_HEADS_UP"):
        directed_067 = []
        if args.require_semantic_sentinels and domain == "TRUE_HEADS_UP":
            directed_067.extend(
                (0, 0, holes, board, "HUSB_RIVER_TP")
                for holes, board in explicit_husb_067_candidates()
            )
            directed_067.extend(
                (0, 1, holes, board, "HUBB_LIMP_TURN_PROBE")
                for holes, board in explicit_hubb_067_candidates()
            )
            directed_067.extend(
                (0, 1, holes, board, "HUBB_VILLAIN_INIT_TURN_PROBE")
                for holes, board in explicit_hubb_villain_init_067_candidates()
            )
        for attempt in range(int(args.max_search)):
            dealer_cycle = 3 if domain == "THREE_HANDED" else 2
            dealer = attempt % dealer_cycle
            opponent_style = "PASSIVE"
            stack_chips = None
            explicit_holes = None
            explicit_board = None

            explicit_label = ""
            if attempt < len(directed_067):
                # Small source-directed fixtures. HUSB probes target the
                # historical river-value branch; HUBB probes target the
                # limp/check/check turn-probe branch.
                (
                    dealer,
                    hero_seat,
                    explicit_holes,
                    explicit_board,
                    explicit_label,
                ) = directed_067[attempt]
                stack_chips = 750
                if explicit_label == "HUBB_VILLAIN_INIT_TURN_PROBE":
                    opponent_style = "PREFLOP_MIN_THEN_PASSIVE"
            else:
                if args.require_semantic_sentinels:
                    # Sentinel search deliberately spans both deep and shallow
                    # geometries plus passive/aggressive histories. This is a
                    # deterministic fixture-discovery lane, not benchmark sampling.
                    contexts = (
                        (500 if domain == "THREE_HANDED" else 750, "PASSIVE"),
                        (240, "PASSIVE"),
                        (160, "PASSIVE"),
                        (120, "PASSIVE"),
                        (500 if domain == "THREE_HANDED" else 750, "MIN_AGGRO"),
                        (240, "MIN_AGGRO"),
                        (160, "MIN_AGGRO"),
                        (120, "ALLIN_PRESSURE"),
                    )
                    context_index = (attempt // dealer_cycle) % len(contexts)
                    stack_chips, opponent_style = contexts[context_index]
                episode_probe = make_episode(
                    domain,
                    dealer,
                    stack_chips=stack_chips,
                )
                live_probe = [
                    seat for seat, stack in enumerate(episode_probe.stacks)
                    if stack > 0
                ]
                # Cycle the DeepCrusher seat so position is not accidentally fixed.
                hero_seat = live_probe[
                    (attempt // (
                        dealer_cycle
                        * (8 if args.require_semantic_sentinels else 1)
                    ))
                    % len(live_probe)
                ]

            episode = make_episode(
                domain,
                dealer,
                stack_chips=stack_chips,
            )
            deal_seed = int(
                args.seed
                + 100000 * (0 if domain == "THREE_HANDED" else 1)
                + attempt
            )

            deal, actions, hero_streets = play_candidate(
                solver,
                policy,
                episode,
                hero_seat=hero_seat,
                deal_seed=deal_seed,
                case_index=attempt,
                opponent_style=opponent_style,
                explicit_holes=explicit_holes,
                explicit_board=explicit_board,
            )
            hero_rows = [a for a in actions if a.hero]
            if explicit_holes is not None:
                trace_rows = [
                    {
                        "street": int(a.street),
                        "family": str(a.decision_family),
                        "action": ACTION_TEXT[int(a.action_type)],
                        "amount_to": int(a.amount_to),
                        "hand_class": (
                            None
                            if a.decision_metadata is None
                            else a.decision_metadata.get("hand_class")
                        ),
                        "return_value": (
                            None
                            if a.decision_metadata is None
                            else a.decision_metadata.get("openppl_return_value")
                        ),
                        "action_name": (
                            None
                            if a.decision_metadata is None
                            else a.decision_metadata.get("openppl_action_name")
                        ),
                        "user_flags_067": (
                            []
                            if a.decision_metadata is None
                            else [
                                name
                                for name in a.decision_metadata.get(
                                    "openppl_user_variables", []
                                )
                                if str(name).lower() in {
                                    "user_turnprobeopportunity",
                                    "user_river66",
                                    "user_rivercbet60",
                                    "user_flop66",
                                    "user_turn66",
                                    "user_66turn75orguriver",
                                    "user_66turn75river",
                                    "user_diddonkonturn",
                                    "user_rivervaluedb",
                                }
                            ]
                        ),
                        "sizing_trace": (
                            []
                            if a.decision_metadata is None
                            else [
                                row
                                for row in a.decision_metadata.get(
                                    "openppl_main_trace", []
                                )
                                if "betsize" in str(row.get("function", "")).lower()
                                or str(row.get("action_name", "")).lower()
                                in {
                                    "bettwothirdpot",
                                    "bethalfpot",
                                    "betthreefourthpot",
                                }
                            ]
                        ),
                    }
                    for a in hero_rows
                ]
                print(
                    "OPENHOLDEM_DIRECTED_067_TRACE "
                    f"attempt={attempt} hero={card_text(deal.holes[hero_seat][0])}"
                    f"{card_text(deal.holes[hero_seat][1])} "
                    f"board={''.join(card_text(x) for x in deal.board)} "
                    f"label={explicit_label} "
                    f"rows={json.dumps(trace_rows, sort_keys=True)}",
                    flush=True,
                )
            hero_action_letters = {
                ACTION_TEXT[int(a.action_type)] for a in hero_rows
            }
            hero_decision_families = {
                str(a.decision_family) for a in hero_rows
            }
            is_full_street = hero_streets == {0, 1, 2, 3}

            adds_action = bool(hero_action_letters - covered_actions)
            adds_street = bool(hero_streets - covered_streets[domain])
            new_semantic_sentinels = (
                (hero_decision_families & required_decision_families)
                - covered_decision_families
            )
            adds_semantic_sentinel = bool(new_semantic_sentinels)
            needs_full_street = (
                is_full_street
                and full_street_counts[domain] < int(args.cases_per_domain)
            )
            if not (
                adds_action
                or adds_street
                or needs_full_street
                or adds_semantic_sentinel
            ):
                continue

            if new_semantic_sentinels:
                for family in sorted(new_semantic_sentinels):
                    print(
                        "OPENHOLDEM_SEMANTIC_SENTINEL_FOUND "
                        f"family={family} domain={domain} attempt={attempt} "
                        f"deal_seed={deal_seed} stack={stack_chips or 'default'} "
                        f"opponent_style={opponent_style} "
                        f"explicit={explicit_holes is not None}",
                        flush=True,
                    )

            serial = case_serial[domain]
            case_serial[domain] += 1
            case_id = f"dc0_{domain.lower()}_{serial:02d}"
            text = render_case(
                case_id=case_id,
                episode=episode,
                deal=deal,
                actions=actions,
                hero_seat=hero_seat,
            )
            path = args.output_dir / f"{case_id}.txt"
            path.write_text(text, encoding="utf-8")
            hero_actions = [
                {
                    "street": int(a.street),
                    "action": ACTION_TEXT[int(a.action_type)],
                    "amount_to": int(a.amount_to),
                    "legal_buttons": a.legal_buttons,
                    "decision_family": a.decision_family,
                    "decision_metadata": a.decision_metadata,
                }
                for a in hero_rows
            ]
            manifest["cases"].append(
                {
                    "id": case_id,
                    "file": path.name,
                    "domain": domain,
                    "dealer": int(episode.dealer_id),
                    "hero_seat": int(hero_seat),
                    "deal_seed": int(deal_seed),
                    "stack_chips": int(max(episode.stacks)),
                    "opponent_style": opponent_style,
                    "explicit_deal": bool(explicit_holes is not None),
                    "explicit_label": explicit_label,
                    "full_street_case": bool(is_full_street),
                    "hero_actions": hero_actions,
                }
            )
            covered_actions.update(hero_action_letters)
            covered_decision_families.update(hero_decision_families)
            covered_streets[domain].update(hero_streets)
            if needs_full_street:
                full_street_counts[domain] += 1

            semantic_done = (
                not required_decision_families
                or required_decision_families <= covered_decision_families
            )
            if (
                full_street_counts[domain] >= int(args.cases_per_domain)
                and covered_streets[domain] == {0, 1, 2, 3}
                and (
                    domain != "TRUE_HEADS_UP"
                    or covered_actions >= required_actions
                )
                and (
                    not args.require_semantic_sentinels
                    or semantic_done
                )
            ):
                break

    # Recompute from serialized hero actions so the manifest remains the single
    # evidence carrier even if RenderedAction storage changes later.
    covered_decision_families_serialized = sorted({
        str(action.get("decision_family", "UNKNOWN"))
        for case in manifest["cases"]
        for action in case.get("hero_actions", [])
    })

    missing_actions = sorted(required_actions - covered_actions)
    missing_streets = {
        domain: sorted({0, 1, 2, 3} - values)
        for domain, values in covered_streets.items()
        if values != {0, 1, 2, 3}
    }
    missing_full = {
        domain: int(args.cases_per_domain) - count
        for domain, count in full_street_counts.items()
        if count < int(args.cases_per_domain)
    }
    missing_semantic = sorted(
        required_decision_families - covered_decision_families
    )
    if missing_actions or missing_streets or missing_full or missing_semantic:
        raise RuntimeError(
            "parity pack search incomplete: "
            f"missing_actions={missing_actions} "
            f"missing_streets={missing_streets} "
            f"missing_full_street_cases={missing_full} "
            f"missing_semantic_sentinels={missing_semantic}"
        )

    manifest["coverage"] = {
        "required_action_buttons": sorted(required_actions),
        "covered_action_buttons": sorted(covered_actions),
        "street_ids_by_domain": {
            domain: sorted(values)
            for domain, values in covered_streets.items()
        },
        "full_street_cases_by_domain": dict(full_street_counts),
        "decision_families": covered_decision_families_serialized,
        "required_decision_families": sorted(required_decision_families),
        "semantic_sentinels": {
            "allin_adjusted_present": "ALLIN_ADJUSTED" in covered_decision_families,
            "tech_betpot_033_present": "TECH_BETPOT_033" in covered_decision_families,
            "tech_betpot_067_present": "TECH_BETPOT_067" in covered_decision_families,
        },
        "note": (
            "R means exact NL raise/betsize is also checked by TestSuite2; "
            "F/C/K/A cover fold/call/check/all-in buttons."
        ),
    }

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    collection = args.output_dir / "deepcrusher_dc0_parity.tc"
    collection.write_text(
        "".join(case["file"] + "\n" for case in manifest["cases"]),
        encoding="utf-8",
    )

    print(f"cases={len(manifest['cases'])}")
    print(f"manifest={manifest_path.resolve()}")
    print(f"collection={collection.resolve()}")
    print("DEEPC_RUSHER_OPENHOLDEM_PARITY_PACK_GENERATION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
