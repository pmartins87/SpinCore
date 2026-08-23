from __future__ import annotations

import math

import r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot as target


def _observation(*, street: int, events: list[tuple[int, int, int, int]]) -> bytes:
    raw = bytearray(120 + 20 * len(events))
    raw[:8] = b"SPNNIV3\x00"
    raw[9] = int(street)
    raw[116:120] = int(len(events)).to_bytes(4, "little", signed=False)
    for index, (actor, event_street, action, forced) in enumerate(events):
        offset = 120 + 20 * index
        raw[offset + 0] = int(actor)
        raw[offset + 1] = int(event_street)
        raw[offset + 2] = int(action)
        raw[offset + 3] = int(forced)
    return bytes(raw)


class _Native:
    def __init__(self, policy):
        self.policy = tuple(policy)

    def __call__(self, _state, _observation, _legal):
        return self.policy


def _close(a: float, b: float, tol: float = 1e-12) -> None:
    assert math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol), (a, b)


def main() -> int:
    assert target.FLOOR == 0.25
    assert target.CHUNKS_PER_ITERATION == 4
    assert target.ROOTS_PER_CHUNK == 64
    assert target.TOTAL_ROOTS == 768
    assert target.CAUSAL_ABS_MIN == 0.02
    assert target.CAUSAL_REL_MIN == 0.10
    assert target.COMMON_P95_MAX_DEGRADE == 0.02
    assert target.NATIVE_MEAN_MAX_DEGRADE == 0.01

    root = _observation(street=0, events=[(1, 0, 2, 1), (2, 0, 2, 1)])
    street, count = target._v3_street_and_nonforced_preflop(root)
    assert street == 0 and count == 0

    continuation = _observation(
        street=0,
        events=[(1, 0, 2, 1), (2, 0, 2, 1), (0, 0, 4, 0)],
    )
    street, count = target._v3_street_and_nonforced_preflop(continuation)
    assert street == 0 and count == 1

    postflop = _observation(
        street=1,
        events=[(1, 0, 2, 1), (2, 0, 2, 1), (0, 0, 4, 0), (1, 0, 1, 0)],
    )
    street, count = target._v3_street_and_nonforced_preflop(postflop)
    assert street == 1 and count == 2

    legal = (0, 1, 2)
    native = [0.0] * 10
    native[0] = 0.60
    native[1] = 0.30
    native[2] = 0.10
    wrapped = target.PreflopContinuationFloorPolicy(_Native(native))

    root_policy = wrapped(None, root, legal)
    for slot in legal:
        _close(root_policy[slot], native[slot])

    damped = wrapped(None, continuation, legal)
    uniform = 1.0 / 3.0
    for slot in legal:
        _close(damped[slot], 0.75 * native[slot] + 0.25 * uniform)
    _close(sum(damped), 1.0)

    post = wrapped(None, postflop, legal)
    for slot in legal:
        _close(post[slot], native[slot])

    stats = wrapped.stats()
    assert stats["calls"] == 3
    assert stats["damped_calls"] == 1
    assert stats["root_preflop_native_calls"] == 1
    assert stats["postflop_native_calls"] == 1

    broken = bytearray(continuation)
    broken.extend(b"x")
    try:
        target._v3_street_and_nonforced_preflop(bytes(broken))
    except RuntimeError:
        pass
    else:
        raise AssertionError("Phase2B6 parser accepted invalid SPNNIV3 wire length")

    # A uniform native policy remains exactly uniform after the floor. This is
    # the iteration-1 neutrality relied on by the frozen precommit.
    uniform_native = [0.0] * 10
    for slot in legal:
        uniform_native[slot] = uniform
    mixed_uniform = target._mix_uniform(uniform_native, legal)
    for slot in legal:
        _close(mixed_uniform[slot], uniform)

    print("R7.5 architecture-reset Phase2B6 preflop-damping synthetic tests PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
