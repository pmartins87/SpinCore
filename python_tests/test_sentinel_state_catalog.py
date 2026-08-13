import hashlib
import struct

import pytest

from spincore.sentinel_state_catalog import (
    find_sentinel_state,
    hole_class_from_neural_bytes,
    normalize_hand_class,
)


def _observation(card0_id, card1_id, *, legal=(0, 1)):
    cards = bytes([card0_id + 1, card1_id + 1, 0, 0, 0, 0, 0])
    numeric = struct.pack("<16f", *([0.0] * 16))
    categorical = bytes([0] * 8)
    legal_mask = bytes([1 if i in legal else 0 for i in range(6)])
    return b"SPNNIV1\x00" + cards + numeric + categorical + legal_mask + bytes([0]) + bytes(32)


AA = _observation(48, 49)      # As Ah
SEVEN_TWO_OFF = _observation(20, 1)  # 7s 2h
AKS = _observation(48, 44)     # As Ks


class _FakeState:
    def __init__(self, observation, *, actor=0, legal=(0, 1), after=None):
        self._observation = observation
        self.actor = actor
        self._legal = tuple(legal)
        self.terminal = False
        self._after = after or {}
        self.closed = False

    def legal_actions(self):
        return self._legal

    def neural_bytes(self):
        return self._observation

    def apply(self, action):
        nxt = self._after.get(action)
        if nxt is not None:
            self._observation = nxt._observation
            self.actor = nxt.actor
            self._legal = nxt._legal
            self.terminal = nxt.terminal
            self._after = nxt._after
        return self

    def close(self):
        self.closed = True


class _FakeSolver:
    def __init__(self, by_seed):
        self.by_seed = dict(by_seed)
        self.created = []

    def create(self, _episode, seed):
        self.created.append(seed)
        template = self.by_seed[seed]
        return _FakeState(
            template._observation,
            actor=template.actor,
            legal=template._legal,
            after=template._after,
        )


def test_hole_class_decoding_and_normalization():
    assert hole_class_from_neural_bytes(AA) == "AA"
    assert hole_class_from_neural_bytes(SEVEN_TWO_OFF) == "72o"
    assert hole_class_from_neural_bytes(AKS) == "AKs"
    assert normalize_hand_class("kas") == "AKs"
    assert normalize_hand_class("27O") == "72o"
    assert normalize_hand_class("aa") == "AA"
    with pytest.raises(ValueError):
        normalize_hand_class("AK")


def test_locator_returns_first_matching_seed_and_exact_observation_identity():
    solver = _FakeSolver(
        {
            10: _FakeState(SEVEN_TWO_OFF),
            11: _FakeState(AKS),
            12: _FakeState(AA),
            13: _FakeState(AA),
        }
    )
    found = find_sentinel_state(
        solver,
        object(),
        target_hand_class="AA",
        seed_start=10,
        seed_stop=14,
    )
    assert found.deck_seed == 12
    assert found.hand_class == "AA"
    assert found.actor == 0
    assert found.legal_actions == (0, 1)
    assert found.observation == AA
    assert found.observation_sha256 == hashlib.sha256(AA).hexdigest()
    assert solver.created == [10, 11, 12]


def test_locator_can_bind_actor_and_action_prefix():
    after_raise = _FakeState(AA, actor=1, legal=(0, 2, 5))
    root = _FakeState(SEVEN_TWO_OFF, actor=0, legal=(1, 3), after={3: after_raise})
    solver = _FakeSolver({5: root})
    found = find_sentinel_state(
        solver,
        object(),
        target_hand_class="AA",
        target_actor=1,
        action_prefix=(3,),
        seed_start=5,
        seed_stop=6,
    )
    assert found.deck_seed == 5
    assert found.actor == 1
    assert found.action_prefix == (3,)
    assert found.legal_actions == (0, 2, 5)


def test_invalid_prefix_fails_without_scanning_other_seeds():
    solver = _FakeSolver(
        {
            1: _FakeState(AA, legal=(0, 1)),
            2: _FakeState(AA, legal=(0, 1)),
        }
    )
    with pytest.raises(ValueError, match="invalid sentinel action_prefix"):
        find_sentinel_state(
            solver,
            object(),
            target_hand_class="AA",
            action_prefix=(5,),
            seed_start=1,
            seed_stop=3,
        )
    assert solver.created == [1]


def test_missing_hand_class_fails_closed():
    solver = _FakeSolver({1: _FakeState(AKS), 2: _FakeState(AKS)})
    with pytest.raises(LookupError, match="no AA sentinel state"):
        find_sentinel_state(
            solver,
            object(),
            target_hand_class="AA",
            seed_start=1,
            seed_stop=3,
        )
