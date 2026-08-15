from __future__ import annotations

from pathlib import Path

from spincore.solver import Episode, SolverLibrary
from spincore.solver_v3 import neural_bytes_v3
from spincore_nn.codec_v3 import decode_spnniv3


def test_solver_v3_bridge_reads_authoritative_variable_length_payload() -> None:
    library = Path("build/libspincore_solver_c.so")
    if not library.exists():
        raise AssertionError("main regression must build solver before Python tests")
    solver = SolverLibrary(library)
    episode = Episode(
        total_chips=1500,
        game_is_hu=False,
        blind_index=0,
        small_blind=10,
        big_blind=20,
        stacks=(500, 500, 500),
        dealer_id=0,
    )
    state = solver.create(episode, 7532026)
    try:
        payload = neural_bytes_v3(state)
        decoded = decode_spnniv3(payload)
        assert payload.startswith(b"SPNNIV3\0")
        assert decoded.history_len == 2  # forced SB + BB
        assert decoded.categorical[0] == 0
        assert decoded.categorical[1] == 0
        assert all(2 <= rank <= 14 for rank in decoded.rank_tokens[:2])
    finally:
        state.close()
