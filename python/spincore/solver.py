from __future__ import annotations

from dataclasses import dataclass
import ctypes as C
import math
from pathlib import Path
from typing import Iterator, Sequence


@dataclass(frozen=True)
class Episode:
    """Tournament state needed to create an authoritative solver root.

    ``dead_players`` preserves elimination order when a 3-handed Spin has
    already become true heads-up.  For the common case where it is omitted,
    zero-stack seats are inferred for HU roots.
    """

    total_chips: int
    game_is_hu: bool
    blind_index: int
    small_blind: int
    big_blind: int
    stacks: tuple[int, int, int]
    dealer_id: int
    dead_players: tuple[int, ...] = ()


class _ScenarioV2(C.Structure):
    _fields_ = [
        ("total_chips", C.c_int32),
        ("game_is_hu", C.c_int32),
        ("blind_index", C.c_int32),
        ("small_blind", C.c_int32),
        ("big_blind", C.c_int32),
        ("stack_0", C.c_int32),
        ("stack_1", C.c_int32),
        ("stack_2", C.c_int32),
        ("dead_player_0", C.c_int32),
        ("dead_player_1", C.c_int32),
        ("dead_player_count", C.c_int32),
        ("dealer_id", C.c_int32),
    ]


def _validate_episode(e: Episode) -> tuple[int, ...]:
    if len(e.stacks) != 3:
        raise ValueError("SpinCore episode must contain exactly three seat stacks")
    if e.total_chips <= 0 or e.small_blind <= 0 or e.big_blind <= 0:
        raise ValueError("invalid tournament/blind values")
    if e.small_blind >= e.big_blind:
        raise ValueError("small blind must be smaller than big blind")
    if e.blind_index < 0:
        raise ValueError("blind_index must be non-negative")
    if e.dealer_id not in (0, 1, 2):
        raise ValueError("dealer_id outside seat range")
    if any(int(x) < 0 for x in e.stacks):
        raise ValueError("negative stack in episode")

    dead = tuple(int(x) for x in e.dead_players)
    if not dead and e.game_is_hu:
        dead = tuple(i for i, stack in enumerate(e.stacks) if int(stack) <= 0)
    if len(dead) > 2:
        raise ValueError("too many dead players for ABI v2")
    if len(set(dead)) != len(dead) or any(x not in (0, 1, 2) for x in dead):
        raise ValueError("invalid dead_players sequence")
    return dead


class SolverLibrary:
    """ctypes owner for the canonical ``SPINCORE_SOLVER_C_ABI_V2`` API."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.lib = C.CDLL(str(self.path))
        L = self.lib

        L.spincore_solver_c_abi_version.argtypes = []
        L.spincore_solver_c_abi_version.restype = C.c_int32
        L.spincore_solver_last_error.argtypes = []
        L.spincore_solver_last_error.restype = C.c_char_p
        if int(L.spincore_solver_c_abi_version()) != 2:
            raise RuntimeError("SPINCORE_SOLVER_C_ABI_V2 required")

        L.spincore_solver_state_create_v2.argtypes = [C.POINTER(_ScenarioV2), C.c_uint64]
        L.spincore_solver_state_create_v2.restype = C.c_void_p
        L.spincore_solver_state_clone.argtypes = [C.c_void_p]
        L.spincore_solver_state_clone.restype = C.c_void_p
        L.spincore_solver_state_destroy.argtypes = [C.c_void_p]
        L.spincore_solver_state_destroy.restype = None

        L.spincore_solver_state_terminal.argtypes = [C.c_void_p]
        L.spincore_solver_state_terminal.restype = C.c_int32
        L.spincore_solver_state_actor.argtypes = [C.c_void_p]
        L.spincore_solver_state_actor.restype = C.c_int32
        L.spincore_solver_state_domain.argtypes = [C.c_void_p]
        L.spincore_solver_state_domain.restype = C.c_int32
        L.spincore_solver_state_legal_mask.argtypes = [C.c_void_p]
        L.spincore_solver_state_legal_mask.restype = C.c_uint32
        L.spincore_solver_state_apply_abstract.argtypes = [C.c_void_p, C.c_int32]
        L.spincore_solver_state_apply_abstract.restype = C.c_int32

        L.spincore_solver_state_neural_input.argtypes = [
            C.c_void_p,
            C.POINTER(C.c_uint8),
            C.c_size_t,
        ]
        L.spincore_solver_state_neural_input.restype = C.c_size_t

        L.spincore_solver_state_terminal_chip_delta.argtypes = [
            C.c_void_p,
            C.POINTER(C.c_int32),
        ]
        L.spincore_solver_state_terminal_chip_delta.restype = C.c_int32
        L.spincore_solver_state_terminal_icm_delta.argtypes = [
            C.c_void_p,
            C.POINTER(C.c_double),
            C.POINTER(C.c_double),
        ]
        L.spincore_solver_state_terminal_icm_delta.restype = C.c_int32

        L.spincore_solver_frontier_create_until_actor.argtypes = [
            C.c_void_p,
            C.c_int32,
            C.c_size_t,
            C.c_size_t,
        ]
        L.spincore_solver_frontier_create_until_actor.restype = C.c_void_p
        L.spincore_solver_frontier_destroy.argtypes = [C.c_void_p]
        L.spincore_solver_frontier_destroy.restype = None
        L.spincore_solver_frontier_size.argtypes = [C.c_void_p]
        L.spincore_solver_frontier_size.restype = C.c_size_t
        L.spincore_solver_frontier_nodes_visited.argtypes = [C.c_void_p]
        L.spincore_solver_frontier_nodes_visited.restype = C.c_size_t
        L.spincore_solver_frontier_max_depth_reached.argtypes = [C.c_void_p]
        L.spincore_solver_frontier_max_depth_reached.restype = C.c_size_t
        L.spincore_solver_frontier_is_terminal.argtypes = [C.c_void_p, C.c_size_t]
        L.spincore_solver_frontier_is_terminal.restype = C.c_int32
        L.spincore_solver_frontier_clone_state.argtypes = [C.c_void_p, C.c_size_t]
        L.spincore_solver_frontier_clone_state.restype = C.c_void_p

    def error(self) -> str:
        return (self.lib.spincore_solver_last_error() or b"").decode("utf-8", "replace")

    def _scenario(self, e: Episode) -> _ScenarioV2:
        dead = _validate_episode(e)
        return _ScenarioV2(
            int(e.total_chips),
            int(bool(e.game_is_hu)),
            int(e.blind_index),
            int(e.small_blind),
            int(e.big_blind),
            int(e.stacks[0]),
            int(e.stacks[1]),
            int(e.stacks[2]),
            int(dead[0]) if len(dead) >= 1 else -1,
            int(dead[1]) if len(dead) >= 2 else -1,
            len(dead),
            int(e.dealer_id),
        )

    def create(self, e: Episode, seed: int) -> "SolverState":
        scenario = self._scenario(e)
        ptr = self.lib.spincore_solver_state_create_v2(C.byref(scenario), C.c_uint64(int(seed)))
        if not ptr:
            raise RuntimeError(self.error() or "solver state creation failed")
        return SolverState(self, ptr)


class SolverState:
    def __init__(self, owner: SolverLibrary, ptr: int | C.c_void_p):
        self.owner = owner
        self.ptr = ptr if isinstance(ptr, C.c_void_p) else C.c_void_p(ptr)

    def _require_open(self) -> C.c_void_p:
        if not self.ptr or not self.ptr.value:
            raise RuntimeError("solver state is closed")
        return self.ptr

    def close(self) -> None:
        if self.ptr and self.ptr.value:
            self.owner.lib.spincore_solver_state_destroy(self.ptr)
            self.ptr = C.c_void_p()

    def __enter__(self) -> "SolverState":
        self._require_open()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def clone(self) -> "SolverState":
        ptr = self.owner.lib.spincore_solver_state_clone(self._require_open())
        if not ptr:
            raise RuntimeError(self.owner.error() or "state clone failed")
        return SolverState(self.owner, ptr)

    @property
    def terminal(self) -> bool:
        return bool(self.owner.lib.spincore_solver_state_terminal(self._require_open()))

    @property
    def actor(self) -> int:
        return int(self.owner.lib.spincore_solver_state_actor(self._require_open()))

    @property
    def domain(self) -> int:
        """0 = THREE_HANDED domain, 1 = TRUE_HEADS_UP domain."""
        return int(self.owner.lib.spincore_solver_state_domain(self._require_open()))

    def legal_actions(self) -> tuple[int, ...]:
        mask = int(self.owner.lib.spincore_solver_state_legal_mask(self._require_open()))
        return tuple(i for i in range(6) if mask & (1 << i))

    def apply(self, action: int) -> "SolverState":
        rc = int(self.owner.lib.spincore_solver_state_apply_abstract(self._require_open(), int(action)))
        if rc != 0:
            raise RuntimeError(self.owner.error() or "abstract action failed")
        return self

    def child(self, action: int) -> "SolverState":
        child = self.clone()
        try:
            return child.apply(action)
        except Exception:
            child.close()
            raise

    def neural_bytes(self) -> bytes:
        ptr = self._require_open()
        n = int(self.owner.lib.spincore_solver_state_neural_input(ptr, None, 0))
        if n <= 0:
            raise RuntimeError(self.owner.error() or "state has no neural input")
        buf = (C.c_uint8 * n)()
        got = int(self.owner.lib.spincore_solver_state_neural_input(ptr, buf, n))
        if got != n:
            raise RuntimeError(self.owner.error() or f"neural input size mismatch: {got} != {n}")
        return bytes(buf)

    def terminal_chip_delta(self) -> tuple[int, int, int]:
        out = (C.c_int32 * 3)()
        rc = int(self.owner.lib.spincore_solver_state_terminal_chip_delta(self._require_open(), out))
        if rc != 0:
            raise RuntimeError(self.owner.error() or "terminal chip delta failed")
        return int(out[0]), int(out[1]), int(out[2])

    def terminal_icm_delta(self, payout_by_place: Sequence[float]) -> tuple[float, float, float]:
        if len(payout_by_place) != 3:
            raise ValueError("payout_by_place must be [1st, 2nd, 3rd]")
        payout = tuple(float(x) for x in payout_by_place)
        if any(not math.isfinite(x) for x in payout):
            raise ValueError("non-finite payout")
        inp = (C.c_double * 3)(*payout)
        out = (C.c_double * 3)()
        rc = int(self.owner.lib.spincore_solver_state_terminal_icm_delta(self._require_open(), inp, out))
        if rc != 0:
            raise RuntimeError(self.owner.error() or "terminal ICM delta failed")
        return float(out[0]), float(out[1]), float(out[2])

    def frontier_until_actor(
        self,
        target_actor: int,
        *,
        max_nodes: int = 100_000,
        max_depth: int = 64,
    ) -> "SolverFrontier":
        ptr = self.owner.lib.spincore_solver_frontier_create_until_actor(
            self._require_open(), int(target_actor), int(max_nodes), int(max_depth)
        )
        if not ptr:
            raise RuntimeError(self.owner.error() or "native frontier creation failed")
        return SolverFrontier(self.owner, ptr)


class SolverFrontier:
    """Owning wrapper for the native R7.1 own-reach frontier."""

    def __init__(self, owner: SolverLibrary, ptr: int | C.c_void_p):
        self.owner = owner
        self.ptr = ptr if isinstance(ptr, C.c_void_p) else C.c_void_p(ptr)

    def _require_open(self) -> C.c_void_p:
        if not self.ptr or not self.ptr.value:
            raise RuntimeError("solver frontier is closed")
        return self.ptr

    def close(self) -> None:
        if self.ptr and self.ptr.value:
            self.owner.lib.spincore_solver_frontier_destroy(self.ptr)
            self.ptr = C.c_void_p()

    def __enter__(self) -> "SolverFrontier":
        self._require_open()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return int(self.owner.lib.spincore_solver_frontier_size(self._require_open()))

    @property
    def nodes_visited(self) -> int:
        return int(self.owner.lib.spincore_solver_frontier_nodes_visited(self._require_open()))

    @property
    def max_depth_reached(self) -> int:
        return int(self.owner.lib.spincore_solver_frontier_max_depth_reached(self._require_open()))

    def is_terminal(self, index: int) -> bool:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        rc = int(self.owner.lib.spincore_solver_frontier_is_terminal(self._require_open(), int(index)))
        if rc not in (0, 1):
            raise RuntimeError(self.owner.error() or "frontier terminal query failed")
        return bool(rc)

    def clone_state(self, index: int) -> SolverState:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        ptr = self.owner.lib.spincore_solver_frontier_clone_state(self._require_open(), int(index))
        if not ptr:
            raise RuntimeError(self.owner.error() or "frontier state clone failed")
        return SolverState(self.owner, ptr)

    def cloned_states(self) -> Iterator[SolverState]:
        for index in range(len(self)):
            yield self.clone_state(index)
