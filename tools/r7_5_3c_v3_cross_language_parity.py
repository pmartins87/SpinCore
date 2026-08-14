from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

from spincore_nn.codec_v3 import FIXED_BYTES, HISTORY_EVENT_BYTES, decode_spnniv3


class ScenarioV2(ctypes.Structure):
    _fields_ = [
        ("total_chips", ctypes.c_int32),
        ("game_is_hu", ctypes.c_int32),
        ("blind_index", ctypes.c_int32),
        ("small_blind", ctypes.c_int32),
        ("big_blind", ctypes.c_int32),
        ("stack_0", ctypes.c_int32),
        ("stack_1", ctypes.c_int32),
        ("stack_2", ctypes.c_int32),
        ("dead_player_0", ctypes.c_int32),
        ("dead_player_1", ctypes.c_int32),
        ("dead_player_count", ctypes.c_int32),
        ("dealer_id", ctypes.c_int32),
    ]


def bind(path: Path):
    lib = ctypes.CDLL(str(path))
    lib.spincore_solver_c_abi_version.argtypes = []
    lib.spincore_solver_c_abi_version.restype = ctypes.c_int32
    lib.spincore_solver_last_error.argtypes = []
    lib.spincore_solver_last_error.restype = ctypes.c_char_p
    lib.spincore_solver_state_create_v2.argtypes = [ctypes.POINTER(ScenarioV2), ctypes.c_uint64]
    lib.spincore_solver_state_create_v2.restype = ctypes.c_void_p
    lib.spincore_solver_state_destroy.argtypes = [ctypes.c_void_p]
    lib.spincore_solver_state_destroy.restype = None
    lib.spincore_solver_state_apply_abstract.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.spincore_solver_state_apply_abstract.restype = ctypes.c_int32
    lib.spincore_solver_state_neural_input_v3.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    lib.spincore_solver_state_neural_input_v3.restype = ctypes.c_size_t
    return lib


def last_error(lib) -> str:
    raw = lib.spincore_solver_last_error()
    return raw.decode("utf-8", errors="replace") if raw else ""


def create(lib, scenario: ScenarioV2, seed: int):
    state = lib.spincore_solver_state_create_v2(ctypes.byref(scenario), ctypes.c_uint64(seed))
    if not state:
        raise RuntimeError(f"state_create_v2 failed: {last_error(lib)}")
    return state


def read_v3(lib, state):
    required = int(lib.spincore_solver_state_neural_input_v3(state, None, 0))
    if required <= 0:
        raise RuntimeError(f"V3 size query failed: {last_error(lib)}")
    buffer = (ctypes.c_uint8 * required)()
    written = int(lib.spincore_solver_state_neural_input_v3(state, buffer, required))
    if written != required:
        raise RuntimeError(
            f"V3 copy failed written={written} required={required}: {last_error(lib)}"
        )
    payload = bytes(buffer)
    decoded = decode_spnniv3(payload)
    assert len(payload) == FIXED_BYTES + decoded.history_len * HISTORY_EVENT_BYTES
    return payload, decoded


def apply_abstract(lib, state, action: int) -> None:
    rc = int(lib.spincore_solver_state_apply_abstract(state, int(action)))
    if rc != 0:
        raise RuntimeError(f"apply_abstract({action}) failed: {last_error(lib)}")


def three_handed_roundtrip(lib) -> None:
    scenario = ScenarioV2(
        1500, 0, 3, 10, 20,
        500, 500, 500,
        -1, -1, 0, 0,
    )
    state = create(lib, scenario, 123456)
    try:
        payload, item = read_v3(lib, state)
        assert len(payload) == 160
        assert item.categorical[0] == 0
        assert item.categorical[1] == 0
        assert item.categorical[5] == 3
        assert item.categorical[6] == 0
        assert item.history_len == 2
        assert [event.categorical[3] for event in item.history] == [1, 1]
        assert item.numeric[13] == 3.0

        # Limp / complete / check reaches a real flop without any Python-side
        # reconstruction of cards or history. The C++ carrier must report all
        # five public events and three visible flop ranks.
        apply_abstract(lib, state, 1)
        apply_abstract(lib, state, 1)
        apply_abstract(lib, state, 1)
        payload, flop = read_v3(lib, state)
        assert flop.categorical[1] == 1
        assert flop.categorical[6] == 3
        assert flop.history_len == 5
        assert len(payload) == 220
        assert all(2 <= rank <= 14 for rank in flop.rank_tokens[:5])
        assert flop.rank_tokens[5:] == (0, 0)
        assert [event.categorical[3] for event in flop.history[:2]] == [1, 1]
        assert all(event.categorical[3] == 0 for event in flop.history[2:])
    finally:
        lib.spincore_solver_state_destroy(state)


def heads_up_dead_seat_roundtrip(lib) -> None:
    scenario = ScenarioV2(
        1500, 1, 0, 10, 20,
        0, 750, 750,
        0, -1, 1, 2,
    )
    state = create(lib, scenario, 998877)
    try:
        payload, item = read_v3(lib, state)
        assert len(payload) == 160
        assert item.categorical[0] == 1
        assert item.categorical[5] == 2
        # Physical dead seat 0 is actor-relative rel1 before canonicalization;
        # SPNNIV3 must still present [Hero, live opponent, absent].
        assert item.categorical[7:10] == (0, 0, 2)
        assert item.categorical[2] == 0  # dealer Hero
        assert item.categorical[3] == 0  # HU SB Hero
        assert item.categorical[4] == 1  # live opponent BB
        assert item.numeric[5] == 0.0
        assert item.numeric[8] == 0.0
        assert item.numeric[11] == 0.0
        assert all(event.categorical[0] <= 1 for event in item.history)
    finally:
        lib.spincore_solver_state_destroy(state)


def decoder_fails_closed(lib) -> None:
    scenario = ScenarioV2(
        1500, 0, 0, 10, 20,
        500, 500, 500,
        -1, -1, 0, 0,
    )
    state = create(lib, scenario, 7)
    try:
        payload, _ = read_v3(lib, state)
        corrupt = bytearray(payload)
        corrupt[6] = ord("9")
        try:
            decode_spnniv3(corrupt)
        except ValueError:
            pass
        else:
            raise AssertionError("SPNNIV3 decoder accepted corrupt magic")

        try:
            decode_spnniv3(payload[:-1])
        except ValueError:
            pass
        else:
            raise AssertionError("SPNNIV3 decoder accepted truncated payload")
    finally:
        lib.spincore_solver_state_destroy(state)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()
    lib = bind(args.library.resolve())
    assert int(lib.spincore_solver_c_abi_version()) == 2
    three_handed_roundtrip(lib)
    heads_up_dead_seat_roundtrip(lib)
    decoder_fails_closed(lib)
    print("R7.5.3C SPNNIV3 C++/C-API/Python parity PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
