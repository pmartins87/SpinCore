from __future__ import annotations

from types import SimpleNamespace

import r7_5_arch_reset_v1plus_phase2b13_root_iid64_target_training as p


def test_seed_determinism_and_namespace_separation() -> None:
    a = p._chance_seeds(1342191342, 17, 2, 3)
    b = p._chance_seeds(1342191342, 17, 2, 3)
    c = p._chance_seeds(1342191342, 17, 2, 4)
    assert a == b
    assert a != c
    assert a[0] != a[1]
    t1 = p._traversal_seed(1342191342, 17, 2)
    t2 = p._traversal_seed(1342191342, 17, 2)
    t3 = p._traversal_seed(1801739323, 17, 2)
    assert t1 == t2
    assert t1 != t3


def test_mean_targets() -> None:
    rows = []
    for i in range(p.K):
        rows.append(tuple(float(i + j) for j in range(10)))
    mean = p._mean_targets(rows)
    expected_offset = (p.K - 1) / 2.0
    assert len(mean) == 10
    for j, value in enumerate(mean):
        assert value == expected_offset + j


class DummyMemory:
    def __init__(self) -> None:
        self.items = []

    def add(self, item) -> None:
        self.items.append(item)


def _sample(observation: bytes, target, *, iteration: int = 2, weight: float = 2.0):
    return SimpleNamespace(
        observation=observation,
        legal=(1, 1, 1, 0, 0, 0, 0, 0, 0, 0),
        target=tuple(target),
        weight=float(weight),
        iteration=int(iteration),
    )


def test_root_replacement_preserves_add_position_and_contract() -> None:
    delegate = DummyMemory()
    root_obs = b"SPNNIV3-root"
    proxy = p.RootReplacingAdvantageMemory(
        delegate,
        observation=root_obs,
        iteration=2,
        replacement_target=tuple(float(100 + i) for i in range(10)),
        expected_legal_mask=(1, 1, 1, 0, 0, 0, 0, 0, 0, 0),
    )
    before = _sample(b"other-before", tuple(float(i) for i in range(10)))
    root = _sample(root_obs, tuple(float(-i) for i in range(10)))
    after = _sample(b"other-after", tuple(float(i + 20) for i in range(10)))
    proxy.add(before)
    proxy.add(root)
    proxy.add(after)

    assert proxy.replaced == 1
    assert len(delegate.items) == 3
    assert delegate.items[0] is before
    assert delegate.items[2] is after
    replaced = delegate.items[1]
    assert replaced.observation == root_obs
    assert replaced.legal == root.legal
    assert replaced.weight == 2.0
    assert replaced.iteration == 2
    assert replaced.target == tuple(float(100 + i) for i in range(10))
    assert proxy.original_target == root.target
    assert proxy.original_weight == root.weight


def test_duplicate_root_replacement_rejected() -> None:
    delegate = DummyMemory()
    root_obs = b"SPNNIV3-root"
    proxy = p.RootReplacingAdvantageMemory(
        delegate,
        observation=root_obs,
        iteration=2,
        replacement_target=tuple(float(i) for i in range(10)),
        expected_legal_mask=(1, 1, 1, 0, 0, 0, 0, 0, 0, 0),
    )
    proxy.add(_sample(root_obs, tuple(float(i) for i in range(10))))
    try:
        proxy.add(_sample(root_obs, tuple(float(i + 1) for i in range(10))))
    except RuntimeError as exc:
        assert "multiple initial-root" in str(exc)
    else:
        raise AssertionError("duplicate root replacement should fail")


def main() -> int:
    test_seed_determinism_and_namespace_separation()
    test_mean_targets()
    test_root_replacement_preserves_add_position_and_contract()
    test_duplicate_root_replacement_rejected()
    print("R7.5 architecture-reset Phase2B13 root-IID64 synthetic tests PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
