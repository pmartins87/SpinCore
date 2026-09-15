from __future__ import annotations

"""Python bridge for the additive lean-legacy action ABI.

Kept separate from ``solver.py`` so the historical generic/universal solver
contract remains untouched.  The functional path wraps a normal SolverState and
changes only how universal legal actions/children are resolved.
"""

import ctypes as C

from spincore.r7_5_action_cfr import UniversalPartialExactCollector


def _configure(owner) -> None:
    lib = owner.lib
    if getattr(owner, "_lean_action_abi_ready", False):
        return
    try:
        legal = lib.spincore_solver_state_lean_legal_mask
        apply = lib.spincore_solver_state_apply_lean
        resolve = lib.spincore_solver_state_resolve_lean_exact
    except AttributeError as exc:
        raise RuntimeError(
            "solver library predates lean legacy-action ABI; rebuild SpinCore"
        ) from exc
    legal.argtypes = [C.c_void_p, C.c_uint32]
    legal.restype = C.c_uint32
    apply.argtypes = [C.c_void_p, C.c_uint32, C.c_int32]
    apply.restype = C.c_int32
    resolve.argtypes = [
        C.c_void_p,
        C.c_uint32,
        C.c_int32,
        C.POINTER(C.c_int32),
        C.POINTER(C.c_int32),
    ]
    resolve.restype = C.c_int32
    owner._lean_action_abi_ready = True


def _validated_mask(active_mask: int) -> int:
    mask = int(active_mask)
    if mask < 0 or mask > 0x3FF:
        raise ValueError("lean active mask must use only universal slots 0..9")
    return mask


def lean_legal_actions(state, active_mask: int) -> tuple[int, ...]:
    _configure(state.owner)
    mask = _validated_mask(active_mask)
    raw = int(
        state.owner.lib.spincore_solver_state_lean_legal_mask(
            state._p(), C.c_uint32(mask)
        )
    )
    if raw == 0 and not state.terminal:
        error = state.owner.error()
        if error:
            raise RuntimeError(error)
    return tuple(i for i in range(10) if raw & (1 << i))


def apply_lean(state, active_mask: int, action: int):
    _configure(state.owner)
    mask = _validated_mask(active_mask)
    slot = int(action)
    if slot < 0 or slot > 9:
        raise ValueError("bad lean action slot")
    rc = state.owner.lib.spincore_solver_state_apply_lean(
        state._p(), C.c_uint32(mask), C.c_int32(slot)
    )
    if rc != 0:
        raise RuntimeError(state.owner.error() or "lean action apply failed")
    return state


def resolve_lean_exact(state, active_mask: int, action: int) -> tuple[int, int]:
    _configure(state.owner)
    mask = _validated_mask(active_mask)
    slot = int(action)
    out_type = C.c_int32()
    out_amount = C.c_int32()
    rc = state.owner.lib.spincore_solver_state_resolve_lean_exact(
        state._p(),
        C.c_uint32(mask),
        C.c_int32(slot),
        C.byref(out_type),
        C.byref(out_amount),
    )
    if rc != 0:
        raise RuntimeError(state.owner.error() or "lean exact resolution failed")
    return int(out_type.value), int(out_amount.value)


class LeanSolverState:
    """Delegating state whose universal-action methods use legacy lean semantics."""

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def universal_legal_actions(self, active_mask: int) -> tuple[int, ...]:
        return lean_legal_actions(self.inner, active_mask)

    def child_universal(self, active_mask: int, action: int):
        child = self.inner.clone()
        try:
            apply_lean(child, active_mask, action)
            return LeanSolverState(child)
        except Exception:
            child.close()
            raise

    def resolve_lean_exact(self, active_mask: int, action: int) -> tuple[int, int]:
        return resolve_lean_exact(self.inner, active_mask, action)

    def close(self) -> None:
        self.inner.close()


class LeanLegacyActionCollector(UniversalPartialExactCollector):
    """Existing audited Deep-CFR recursion, legacy-faithful action resolver."""

    @staticmethod
    def _wrap(root):
        return root if isinstance(root, LeanSolverState) else LeanSolverState(root)

    def collect_advantage_partial_exact(
        self,
        root,
        *,
        traverser: int,
        iteration: int,
        exact_opponent_levels: int,
    ):
        return super().collect_advantage_partial_exact(
            self._wrap(root),
            traverser=traverser,
            iteration=iteration,
            exact_opponent_levels=exact_opponent_levels,
        )

    def collect_strategy_own_reach(self, root, *, target_player: int, iteration: int) -> int:
        return super().collect_strategy_own_reach(
            self._wrap(root), target_player=target_player, iteration=iteration
        )
