from __future__ import annotations

import math

import torch

import r7_5_arch_reset_v1plus_phase2b9_robust_advantage_regression as p
from spincore.r7_5_representation_v3_stage_contract import side_member_seeds


class _Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.values = torch.nn.Parameter(torch.zeros(10, dtype=torch.float32))

    def forward(self, batch):
        n = int(batch["legal"].shape[0])
        return self.values.unsqueeze(0).expand(n, -1)


def test_huber_formula() -> None:
    model = _Dummy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    legal = torch.zeros((1, 10), dtype=torch.bool)
    legal[0, 0] = True
    legal[0, 1] = True
    batch = {"legal": legal}
    target = torch.zeros((1, 10), dtype=torch.float32)
    target[0, 0] = 0.01
    target[0, 1] = 0.04
    weights = torch.ones(1, dtype=torch.float32)
    loss = p._huber_train_step(model, optimizer, batch, target, weights, beta=0.02)
    expected_small = 0.5 * (0.01 ** 2) / 0.02
    expected_large = 0.04 - 0.5 * 0.02
    expected = 0.5 * (expected_small + expected_large)
    assert abs(loss - expected) < 1e-7, (loss, expected)


def test_member_seed_contract() -> None:
    for training_seed in (1342191342, 1801739323):
        seen = set()
        for member in range(4):
            init_seed, batch_seed = p._member_seeds(training_seed, member)
            assert isinstance(init_seed, int) and isinstance(batch_seed, int)
            assert (init_seed, batch_seed) not in seen
            seen.add((init_seed, batch_seed))
            if member > 0:
                assert (init_seed, batch_seed) == side_member_seeds(training_seed, 3, member)


def test_tv_and_metric() -> None:
    left = [[1.0, 0.0] + [0.0] * 8, [0.5, 0.5] + [0.0] * 8]
    right = [[0.0, 1.0] + [0.0] * 8, [0.75, 0.25] + [0.0] * 8]
    tv = p._tv_rows(left, right)
    assert len(tv) == 2
    assert abs(tv[0] - 1.0) < 1e-12
    assert abs(tv[1] - 0.25) < 1e-12
    metric = p._metric(tv, left, right)
    assert metric["count"] == 2
    assert abs(metric["mean"] - 0.625) < 1e-12
    assert abs(metric["dominant_action_mismatch_rate"] - 0.5) < 1e-12


def test_pooled_region() -> None:
    evaluations = {
        "2029384436": {
            "MSE_PAIRED_CONTROL": {"regions": {"PREFLOP_ROOT": {"count": 2, "mean": 0.2}}},
            "HUBER_BETA_002": {"regions": {"PREFLOP_ROOT": {"count": 2, "mean": 0.1}}},
        },
        "1150634112": {
            "MSE_PAIRED_CONTROL": {"regions": {"PREFLOP_ROOT": {"count": 3, "mean": 0.4}}},
            "HUBER_BETA_002": {"regions": {"PREFLOP_ROOT": {"count": 3, "mean": 0.3}}},
        },
    }
    row = p._pooled_region(evaluations, "MSE_PAIRED_CONTROL", ("PREFLOP_ROOT",))
    assert row["count"] == 5
    assert abs(row["mean"] - 0.32) < 1e-12


def test_frozen_contract() -> None:
    assert p.HUBER_BETA == 0.02
    assert p.POLICY_COUNT == 1024
    assert p.ABS_IMPROVEMENT_MIN == 0.03
    assert p.REL_IMPROVEMENT_MIN == 0.10
    assert p.P95_MAX_DEGRADE == 0.02
    assert p.REGION_MAX_DEGRADE == 0.01
    assert p.PHASE2B6_RESULT_SHA256 == "33ec6ba89823dae632b7af935def17444379c96a28e59478c0b7c91f1ec3659a"
    assert p.PHASE2B8_RESULT_SHA256 == "1fd9144a488cea6de0a7500320d552abf994908b5200146d4baa4bd6f81c4d98"


def main() -> int:
    torch.set_num_threads(1)
    test_huber_formula()
    test_member_seed_contract()
    test_tv_and_metric()
    test_pooled_region()
    test_frozen_contract()
    print("R7.5 architecture-reset Phase2B9 robust-Advantage synthetic tests PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
