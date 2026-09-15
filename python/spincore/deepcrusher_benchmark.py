from __future__ import annotations

"""Core contracts for the offline SpinCore-vs-DeepCrusher benchmark.

This module deliberately separates *match fairness* from the still-being-built
DeepCrusher decision oracle.  The benchmark must never force DeepCrusher through
SpinCore's seven-action abstraction: DeepCrusher keeps its own exact OpenPPL bet
sizes and those exact actions are applied directly to the SpinCore simulator.

Nothing here connects to a live poker client.  It is an offline simulator/test
contract only.
"""

from dataclasses import dataclass
import ctypes as C
import hashlib
from pathlib import Path
from typing import Iterable, Protocol, Sequence


SPINCORE_POLICY_ID = "SPINCORE"
DEEPC_RUSHER_POLICY_ID = "DEEPCRUSHER"

# Frozen good/stable DeepCrusher baseline from pmartins87/DeepCrusher docs/STATUS.md.
DEEPC_RUSHER_BASELINE_BRANCH = "r8-v22-stable-20260914"
DEEPC_RUSHER_STRATEGIC_SOURCE = "DeepCrusher_R8_v22_CANDIDATE_OPENHOLDEM_RECOVERED_20260914.txt"
DEEPC_RUSHER_STRATEGIC_SHA256 = "9fc2d00aacc915f3c265429f764056f3c6270df616244026aac22e455c803ee9"
DEEPC_RUSHER_OPERATIONAL_SOURCE = "DeepCrusher_R8_v22_CANDIDATE_OPENHOLDEM_ASCII_20260914.txt"
DEEPC_RUSHER_OPERATIONAL_SHA256 = "0113badc99727a7dd47c02448d4d042b5e008534cd63fd79a461a72b24eeb68d"

EXACT_ACTION_NAMES = {
    0: "FOLD",
    1: "CHECK",
    2: "CALL",
    3: "BET_TO",
    4: "RAISE_TO",
    5: "ALL_IN",
}


@dataclass(frozen=True, order=True)
class ExternalExactAction:
    """One exact poker action produced by an external reference strategy."""

    action_type: int
    amount_to: int = 0

    def __post_init__(self) -> None:
        if int(self.action_type) not in EXACT_ACTION_NAMES:
            raise ValueError("action_type must be 0..5")
        if int(self.amount_to) < 0:
            raise ValueError("amount_to must be nonnegative")
        if int(self.action_type) not in (3, 4) and int(self.amount_to) != 0:
            raise ValueError("amount_to is only valid for BET_TO/RAISE_TO")

    @property
    def name(self) -> str:
        return EXACT_ACTION_NAMES[int(self.action_type)]


class OfflineDecisionPolicy(Protocol):
    """Offline policy contract shared by SpinCore and DeepCrusher adapters."""

    policy_id: str

    def choose_exact(self, state, *, seat: int) -> ExternalExactAction:
        ...


@dataclass(frozen=True)
class Lineup:
    """Policy assigned to each logical seat for one exact scenario/deal."""

    seats: tuple[str, str, str]

    def __post_init__(self) -> None:
        if len(self.seats) != 3:
            raise ValueError("lineup needs exactly three logical seats")
        allowed = {SPINCORE_POLICY_ID, DEEPC_RUSHER_POLICY_ID, "DEAD"}
        if any(item not in allowed for item in self.seats):
            raise ValueError(f"unknown policy in lineup: {self.seats}")

    def count(self, policy_id: str) -> int:
        return sum(1 for item in self.seats if item == policy_id)


@dataclass(frozen=True)
class MatchObservation:
    scenario_index: int
    domain: str
    blind: str
    lineup: Lineup
    chip_delta: tuple[int, int, int]

    def __post_init__(self) -> None:
        if len(self.chip_delta) != 3:
            raise ValueError("chip_delta requires three seats")
        if sum(int(x) for x in self.chip_delta) != 0:
            raise ValueError("match observation must be zero-sum")


def balanced_hu_lineups(dead_seat: int) -> tuple[Lineup, Lineup]:
    """Same HU deal twice, swapping SpinCore/DeepCrusher across live seats."""
    if dead_seat not in (0, 1, 2):
        raise ValueError("dead_seat must be 0..2")
    live = [seat for seat in range(3) if seat != dead_seat]
    first = ["DEAD", "DEAD", "DEAD"]
    second = ["DEAD", "DEAD", "DEAD"]
    first[live[0]] = SPINCORE_POLICY_ID
    first[live[1]] = DEEPC_RUSHER_POLICY_ID
    second[live[0]] = DEEPC_RUSHER_POLICY_ID
    second[live[1]] = SPINCORE_POLICY_ID
    return Lineup(tuple(first)), Lineup(tuple(second))


def balanced_three_handed_lineups() -> tuple[Lineup, ...]:
    """Six-game AAB/ABB block with exact seat/composition balance.

    For the same sampled scenario/deal:
      - 3 games contain 2 SpinCore + 1 DeepCrusher, rotating the single
        DeepCrusher through BTN/SB/BB logical seats;
      - 3 games contain 1 SpinCore + 2 DeepCrusher, rotating the single
        SpinCore through every seat.

    Across the complete block each policy occupies every seat exactly three
    times and has nine seat-exposures total.  Therefore no policy receives a
    seat-count or majority-count advantage in the aggregate.
    """
    out: list[Lineup] = []
    for singleton_seat in range(3):
        seats = [SPINCORE_POLICY_ID] * 3
        seats[singleton_seat] = DEEPC_RUSHER_POLICY_ID
        out.append(Lineup(tuple(seats)))
    for singleton_seat in range(3):
        seats = [DEEPC_RUSHER_POLICY_ID] * 3
        seats[singleton_seat] = SPINCORE_POLICY_ID
        out.append(Lineup(tuple(seats)))
    return tuple(out)


def validate_three_handed_balance(lineups: Sequence[Lineup]) -> None:
    if len(lineups) != 6:
        raise ValueError("balanced 3H block requires six lineups")
    for policy in (SPINCORE_POLICY_ID, DEEPC_RUSHER_POLICY_ID):
        if sum(row.count(policy) for row in lineups) != 9:
            raise ValueError(f"unbalanced total exposure for {policy}")
        for seat in range(3):
            if sum(1 for row in lineups if row.seats[seat] == policy) != 3:
                raise ValueError(f"unbalanced seat {seat} exposure for {policy}")


def aggregate_policy_chip_delta(rows: Iterable[MatchObservation]) -> dict[str, int]:
    totals = {SPINCORE_POLICY_ID: 0, DEEPC_RUSHER_POLICY_ID: 0}
    for row in rows:
        for seat, policy in enumerate(row.lineup.seats):
            if policy in totals:
                totals[policy] += int(row.chip_delta[seat])
    if totals[SPINCORE_POLICY_ID] + totals[DEEPC_RUSHER_POLICY_ID] != 0:
        raise ValueError("policy aggregate must remain zero-sum")
    return totals


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_frozen_deepcrusher_source(path: str | Path, *, operational: bool = True) -> dict[str, str]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    expected_name = DEEPC_RUSHER_OPERATIONAL_SOURCE if operational else DEEPC_RUSHER_STRATEGIC_SOURCE
    expected_hash = DEEPC_RUSHER_OPERATIONAL_SHA256 if operational else DEEPC_RUSHER_STRATEGIC_SHA256
    actual = sha256_file(target)
    if target.name != expected_name:
        raise ValueError(f"wrong DeepCrusher baseline filename: {target.name!r}; expected {expected_name!r}")
    if actual.lower() != expected_hash.lower():
        raise ValueError(f"DeepCrusher baseline SHA256 mismatch: {actual}; expected {expected_hash}")
    return {"path": str(target.resolve()), "sha256": actual, "baseline_branch": DEEPC_RUSHER_BASELINE_BRANCH}


def _configure_exact_apply(owner) -> None:
    if getattr(owner, "_external_exact_apply_ready", False):
        return
    try:
        fn = owner.lib.spincore_solver_state_apply_exact
    except AttributeError as exc:
        raise RuntimeError("solver library predates exact external-action benchmark ABI; rebuild SpinCore") from exc
    fn.argtypes = [C.c_void_p, C.c_int32, C.c_int32]
    fn.restype = C.c_int32
    owner._external_exact_apply_ready = True


def apply_external_exact(state, action: ExternalExactAction):
    """Apply a DeepCrusher/reference exact action without quantizing its size."""
    _configure_exact_apply(state.owner)
    rc = state.owner.lib.spincore_solver_state_apply_exact(
        state._p(), C.c_int32(int(action.action_type)), C.c_int32(int(action.amount_to))
    )
    if rc != 0:
        raise RuntimeError(state.owner.error() or "external exact action apply failed")
    return state


def benchmark_contract() -> dict[str, object]:
    validate_three_handed_balance(balanced_three_handed_lineups())
    return {
        "schema": "SPINCORE_VS_DEEPCRUSHER_BENCHMARK_CONTRACT_V1",
        "deepcrusher": {
            "branch": DEEPC_RUSHER_BASELINE_BRANCH,
            "operational_source": DEEPC_RUSHER_OPERATIONAL_SOURCE,
            "operational_sha256": DEEPC_RUSHER_OPERATIONAL_SHA256,
            "strategic_source": DEEPC_RUSHER_STRATEGIC_SOURCE,
            "strategic_sha256": DEEPC_RUSHER_STRATEGIC_SHA256,
        },
        "hu_pairing": "same scenario/deal twice; policies swap live seats",
        "three_handed_pairing": "same scenario/deal six-game AAB/ABB balanced block; each policy occupies each seat exactly three times",
        "primary_hand_metric": "paired empirical-sampler chip EV in chips/hand",
        "secondary_hand_metrics": ["bb/100 by blind", "position", "street", "action family"],
        "ultimate_tournament_metric": "complete SpinGo win rate once continuous blind progression is separately frozen",
        "no_size_quantization": True,
        "offline_only": True,
    }
