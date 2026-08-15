from __future__ import annotations

import ctypes as C


def neural_bytes_v3(state) -> bytes:
    """Read the authoritative variable-length SPNNIV3 payload from SolverState.

    Kept as a versioned bridge instead of mutating the historical SolverState
    V1/V2 methods used by already-frozen experiments.
    """
    state_ptr = state._p()
    lib = state.owner.lib
    fn = getattr(lib, "spincore_solver_state_neural_input_v3", None)
    if fn is None:
        raise RuntimeError("solver library does not expose SPNNIV3 C API")
    fn.argtypes = [C.c_void_p, C.POINTER(C.c_uint8), C.c_size_t]
    fn.restype = C.c_size_t

    required = int(fn(state_ptr, None, 0))
    if required <= 0:
        raise RuntimeError(state.owner.error() or "no SPNNIV3 payload")
    buffer = (C.c_uint8 * required)()
    got = int(fn(state_ptr, buffer, required))
    if got != required:
        raise RuntimeError(
            state.owner.error()
            or f"SPNNIV3 size mismatch: required={required} got={got}"
        )
    return bytes(buffer)
