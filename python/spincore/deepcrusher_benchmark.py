from __future__ import annotations

"""Core contracts and match engine for the offline SpinCore-vs-DeepCrusher benchmark.

The benchmark must never force DeepCrusher through SpinCore's seven-action
abstraction: DeepCrusher keeps its own exact OpenPPL bet sizes and those exact
actions are applied directly to the SpinCore simulator.

Nothing here connects to a live poker client. It is an offline simulator/test
contract only.
"""

from dataclasses import dataclass
import ctypes as C
import hashlib
from pathlib import Path
import random
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
    """One exact poker action produced by an offline benchmark strategy."""

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

    def choose_exact(
        self,
        state,
        *,
        seat: int,
        rng: random.Random,
    ) -> ExternalExactAction:
        ...


class SpinCoreCheckpointPolicy:
    """Expose a trained AveragePolicy through the exact-action benchmark API."""

    policy_id = SPINCORE_POLICY_ID

    def __init__(self, agent) -> None:
        self.agent = agent

    def choose_exact(
        self,
        state,
        *,
        seat: int,
        rng: random.Random,
    ) -> ExternalExactAction:
        del seat  # Agent observation is already actor-relative.
        from spincore.lean_solver_actions import resolve_lean_exact

        active_mask, legal, probs = self.agent.distribution(state)
        x = rng.random()
        cumulative = 0.0
        slot = int(legal[-1])
        for candidate in legal:
            cumulative += float(probs[candidate])
            if x < cumulative:
                slot = int(candidate)
                break
        action_type, amount_to = resolve_lean_exact(state, active_mask, slot)
        return ExternalExactAction(int(action_type), int(amount_to))


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
    decisions: int = 0

    def __post_init__(self) -> None:
        if len(self.chip_delta) != 3:
            raise ValueError("chip_delta requires three seats")
        if sum(int(x) for x in self.chip_delta) != 0:
            raise ValueError("match observation must be zero-sum")
        if int(self.decisions) < 0:
            raise ValueError("decisions must be nonnegative")


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
    """Six-game AAB/ABB block with exact seat/composition balance."""
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


def _mix64(*values: int) -> int:
    x = 0x9E3779B97F4A7C15
    mask = (1 << 64) - 1
    for value in values:
        y = int(value) & mask
        x ^= (y + 0x9E3779B97F4A7C15 + ((x << 6) & mask) + (x >> 2)) & mask
        x &= mask
    return x


class OfflineHeadToHeadEngine:
    """Play exact paired hands between two offline policies on one solver."""

    def __init__(
        self,
        solver_library,
        *,
        spincore_policy: OfflineDecisionPolicy,
        deepcrusher_policy: OfflineDecisionPolicy,
        master_seed: int = 20260915,
        max_decisions: int = 200,
    ) -> None:
        if spincore_policy.policy_id != SPINCORE_POLICY_ID:
            raise ValueError("spincore_policy has wrong policy_id")
        if deepcrusher_policy.policy_id != DEEPC_RUSHER_POLICY_ID:
            raise ValueError("deepcrusher_policy has wrong policy_id")
        self.solver = solver_library
        self.policies = {
            SPINCORE_POLICY_ID: spincore_policy,
            DEEPC_RUSHER_POLICY_ID: deepcrusher_policy,
        }
        self.master_seed = int(master_seed)
        self.max_decisions = int(max_decisions)
        if self.max_decisions <= 0:
            raise ValueError("max_decisions must be positive")

    def play_hand(
        self,
        episode,
        *,
        deal_seed: int,
        scenario_index: int,
        lineup: Lineup,
        lineup_index: int = 0,
    ) -> MatchObservation:
        live = {seat for seat, stack in enumerate(episode.stacks) if int(stack) > 0}
        for seat in range(3):
            if seat in live and lineup.seats[seat] == "DEAD":
                raise ValueError("live episode seat cannot have DEAD lineup policy")
            if seat not in live and lineup.seats[seat] != "DEAD":
                raise ValueError("dead episode seat must have DEAD lineup policy")

        state = self.solver.create(episode, int(deal_seed))
        rngs = {
            seat: random.Random(_mix64(self.master_seed, scenario_index, lineup_index, seat, 0xDCC0))
            for seat in live
        }
        decisions = 0
        try:
            while not state.terminal:
                actor = int(state.actor)
                policy_id = lineup.seats[actor]
                if policy_id == "DEAD":
                    raise RuntimeError("solver selected dead seat as actor")
                policy = self.policies[policy_id]
                action = policy.choose_exact(state, seat=actor, rng=rngs[actor])
                apply_external_exact(state, action)
                decisions += 1
                if decisions > self.max_decisions:
                    raise RuntimeError("benchmark hand exceeded max_decisions")
            delta = tuple(int(x) for x in state.terminal_chip_delta())
            return MatchObservation(
                scenario_index=int(scenario_index),
                domain="TRUE_HEADS_UP" if bool(episode.game_is_hu) else "THREE_HANDED",
                blind=f"{int(episode.small_blind)}/{int(episode.big_blind)}",
                lineup=lineup,
                chip_delta=delta,
                decisions=decisions,
            )
        finally:
            state.close()

    def play_balanced_block(
        self,
        episode,
        *,
        deal_seed: int,
        scenario_index: int,
    ) -> tuple[MatchObservation, ...]:
        if bool(episode.game_is_hu):
            dead = [seat for seat, stack in enumerate(episode.stacks) if int(stack) <= 0]
            if len(dead) != 1:
                raise ValueError("HU episode must contain exactly one dead seat")
            lineups: Sequence[Lineup] = balanced_hu_lineups(dead[0])
        else:
            lineups = balanced_three_handed_lineups()
            validate_three_handed_balance(lineups)
        return tuple(
            self.play_hand(
                episode,
                deal_seed=deal_seed,
                scenario_index=scenario_index,
                lineup=lineup,
                lineup_index=index,
            )
            for index, lineup in enumerate(lineups)
        )


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
