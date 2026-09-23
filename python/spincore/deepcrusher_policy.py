from __future__ import annotations

"""Executable frozen DeepCrusher R8 policy for the offline benchmark."""

from pathlib import Path
import random
import re

from spincore.deepcrusher_action import translate_openppl_decision
from spincore.deepcrusher_benchmark import (
    DEEPC_RUSHER_POLICY_ID,
    DEEPC_RUSHER_OPERATIONAL_SOURCE,
    ExternalExactAction,
    verify_frozen_deepcrusher_source,
)
from spincore.deepcrusher_native_symbols import (
    DeepCrusherPrimitiveSymbols,
    frozen_benchmark_environment,
)
from spincore.deepcrusher_state import (
    STREET_FLOP,
    STREET_PREFLOP,
    STREET_RIVER,
    STREET_TURN,
    state_view,
)
from spincore.openppl_program import OpenPPLProgram, OpenPPLSession


_IDENTIFIER = re.compile(r"(?<![A-Za-z0-9_$])([A-Za-z_][A-Za-z0-9_$]*)(?![A-Za-z0-9_$])")
_MAIN_BY_STREET = {
    STREET_PREFLOP: "f$preflop",
    STREET_FLOP: "f$flop",
    STREET_TURN: "f$turn",
    STREET_RIVER: "f$river",
}
_INIT_STARTUP = "f$ini_function_on_startup"
_INIT_HANDRESET = "f$ini_function_on_handreset"
_INIT_NEW_ROUND = "f$ini_function_on_new_round"
_INIT_MY_TURN = "f$ini_function_on_my_turn"


class DeepCrusherR8Policy:
    """Run the pinned DeepCrusher R8 OpenPPL source as an exact offline policy.

    Benchmark hands are deliberately independent synthetic connections. Each
    DeepCrusher seat gets a fresh OpenPPL session at begin_hand(), preventing
    connection-level memory from one paired replay leaking into its counterpart.
    R8's reserved initialization callbacks are then replayed in OpenHoldem order
    as soon as the first observable decision state exists.
    """

    policy_id = DEEPC_RUSHER_POLICY_ID

    def __init__(
        self,
        *,
        source_path: str | Path,
        library_paths: tuple[str | Path, ...] | list[str | Path],
    ) -> None:
        self.source_path = Path(source_path)
        self.library_paths = tuple(Path(x) for x in library_paths)
        if not self.library_paths:
            raise ValueError("DeepCrusher R8 requires the pinned OpenPPL library")

        self.source_metadata = verify_frozen_deepcrusher_source(
            self.source_path,
            operational=True,
        )
        strategy_text = self.source_path.read_text(encoding="utf-8")
        library_texts = tuple(
            path.read_text(encoding="utf-8") for path in self.library_paths
        )
        self.program = OpenPPLProgram.from_texts(
            strategy_text,
            library_texts=library_texts,
        )

        identifiers = {
            match.group(1)
            for text in (strategy_text, *library_texts)
            for match in _IDENTIFIER.finditer(text)
        }
        self.environment = frozen_benchmark_environment(sorted(identifiers))

        self._sessions: dict[int, OpenPPLSession] = {}
        self._startup_done: set[int] = set()
        self._handreset_done: set[int] = set()
        self._last_street: dict[int, int | None] = {}
        self._big_blind_chips: int | None = None
        self._hand_serial = 0

    @classmethod
    def from_repository(cls, root: str | Path) -> "DeepCrusherR8Policy":
        base = Path(root)
        return cls(
            source_path=(
                base
                / "fixtures"
                / "deepcrusher_r8_v22"
                / DEEPC_RUSHER_OPERATIONAL_SOURCE
            ),
            library_paths=(
                base
                / "fixtures"
                / "openppl_library_dc0"
                / "OpenPPL_Library_part1.ohf",
                base
                / "fixtures"
                / "openppl_library_dc0"
                / "OpenPPL_Library_part2.ohf",
            ),
        )

    def begin_hand(
        self,
        *,
        episode,
        lineup,
        scenario_index: int,
        lineup_index: int,
        deal_seed: int,
    ) -> None:
        """Reset one independent synthetic benchmark hand.

        scenario_index/lineup_index/deal_seed are accepted and intentionally not
        used strategically; keeping them in the hook makes accidental hidden
        dependence visible and testable.
        """
        del scenario_index, lineup_index, deal_seed
        bb = int(episode.big_blind)
        if bb <= 0:
            raise ValueError("episode big blind must be positive")
        self._big_blind_chips = bb
        self._hand_serial += 1

        seats = tuple(
            seat
            for seat, policy in enumerate(lineup.seats)
            if policy == self.policy_id
        )
        self._sessions = {seat: OpenPPLSession(self.program) for seat in seats}
        self._startup_done.clear()
        self._handreset_done.clear()
        self._last_street = {seat: None for seat in seats}

    def _session(self, seat: int) -> OpenPPLSession:
        try:
            return self._sessions[int(seat)]
        except KeyError as exc:
            raise RuntimeError(
                "DeepCrusher choose_exact called before begin_hand or for wrong seat"
            ) from exc

    def _run_lifecycle(
        self,
        *,
        seat: int,
        session: OpenPPLSession,
        provider: DeepCrusherPrimitiveSymbols,
        hand_class: str,
        street: int,
    ) -> None:
        if seat not in self._startup_done:
            if self.program.has_function(_INIT_STARTUP):
                session.run_initialization(
                    _INIT_STARTUP,
                    provider,
                    hand_class=hand_class,
                )
            self._startup_done.add(seat)

        if seat not in self._handreset_done:
            if self.program.has_function(_INIT_HANDRESET):
                session.run_initialization(
                    _INIT_HANDRESET,
                    provider,
                    hand_class=hand_class,
                )
            self._handreset_done.add(seat)

        if self._last_street.get(seat) != int(street):
            if self.program.has_function(_INIT_NEW_ROUND):
                session.run_initialization(
                    _INIT_NEW_ROUND,
                    provider,
                    hand_class=hand_class,
                )
            self._last_street[seat] = int(street)

        if self.program.has_function(_INIT_MY_TURN):
            session.run_initialization(
                _INIT_MY_TURN,
                provider,
                hand_class=hand_class,
            )

    def choose_exact(
        self,
        state,
        *,
        seat: int,
        rng: random.Random,
    ) -> ExternalExactAction:
        del rng  # R8 is deterministic under the frozen no-random benchmark profile.
        if self._big_blind_chips is None:
            raise RuntimeError("DeepCrusher choose_exact called before begin_hand")
        if not getattr(state.owner, "explicit_deal_available", False):
            raise RuntimeError(
                "canonical DeepCrusher benchmark requires explicit deal snapshot ABI "
                "for exact OpenHoldem suit semantics"
            )

        view = state_view(state)
        provider = DeepCrusherPrimitiveSymbols(
            view,
            environment=self.environment,
        )
        hand_class = view.hero_hand_class
        session = self._session(seat)

        self._run_lifecycle(
            seat=int(seat),
            session=session,
            provider=provider,
            hand_class=hand_class,
            street=int(view.street),
        )

        try:
            main = _MAIN_BY_STREET[int(view.street)]
        except KeyError as exc:
            raise RuntimeError(f"unsupported DeepCrusher street: {view.street}") from exc

        decision = session.evaluate(
            main,
            provider,
            hand_class=hand_class,
        )
        public = state.public_snapshot()
        return translate_openppl_decision(
            decision,
            public=public,
            actor=int(seat),
            big_blind_chips=int(self._big_blind_chips),
        )
