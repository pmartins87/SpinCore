from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_r7_3_policy_mixture_uncertainty_damping as accepted
from spincore.r7_5_action_uncertainty import uncertainty_damped_policy_from_advantages
from spincore.solver import Episode, SolverLibrary

LIB = ROOT / "build" / "libspincore_solver_c.so"
EPSILON_SCALE = 1.75
EPSILON_CAP = 0.50


class FixedSix(torch.nn.Module):
    def __init__(self, row):
        super().__init__()
        self.register_buffer("row", torch.tensor(row, dtype=torch.float32))

    def forward(self, batch):
        n = int(batch["numeric"].shape[0])
        return self.row.unsqueeze(0).expand(n, -1)


def episode() -> Episode:
    return Episode(1500, True, 0, 10, 20, (0, 750, 750), 1, (0,))


def _oracle(rows):
    # R7_3_CANDIDATE_SEMANTIC_FREEZE selected size4_uncertainty_s175.
    # The historical runner stores these as module globals; set the selected
    # values explicitly instead of accidentally testing its source defaults.
    accepted.EPSILON_SCALE = EPSILON_SCALE
    accepted.EPSILON_CAP = EPSILON_CAP
    solver = SolverLibrary(LIB)
    state = solver.create(episode(), 123456)
    try:
        observation = state.neural_bytes()
        legal = state.legal_actions()
        behavior = accepted.UncertaintyDampedPolicyMixture(device="cpu")
        behavior.models = [FixedSix(row) for row in rows]
        policy = tuple(float(x) for x in behavior(state, observation, legal))
        stats = {
            "calls": int(behavior.calls),
            "epsilon": float(behavior.epsilon_sum),
            "epsilon_max": float(behavior.epsilon_max),
            "disagreement": float(behavior.disagreement_sum),
            "raw_epsilon": float(behavior.raw_epsilon_max),
            "cap_hit_calls": int(behavior.cap_hit_calls),
            "epsilon_ge_010_calls": int(behavior.epsilon_ge_010_calls),
            "epsilon_ge_025_calls": int(behavior.epsilon_ge_025_calls),
        }
        return policy, legal, stats
    finally:
        state.close()


def _assert_fixture(rows) -> None:
    # The accepted implementation sees float32 network outputs. Quantize once
    # and feed those exact same values to both paths; tolerance remains 1e-12.
    rows_f32 = [
        tuple(float(x) for x in torch.tensor(row, dtype=torch.float32).tolist())
        for row in rows
    ]
    old_policy, legal, old_stats = _oracle(rows_f32)
    widened = [row + (0.0, 0.0, 0.0, 0.0) for row in rows_f32]
    new_policy, new_stats = uncertainty_damped_policy_from_advantages(
        widened,
        tuple(int(x) for x in legal),
        action_count=10,
        epsilon_scale=EPSILON_SCALE,
        epsilon_cap=EPSILON_CAP,
    )
    for action in range(6):
        assert abs(float(new_policy[action]) - float(old_policy[action])) <= 1e-12
    assert tuple(float(x) for x in new_policy[6:]) == (0.0, 0.0, 0.0, 0.0)
    assert old_stats["calls"] == 1
    assert abs(float(new_stats["epsilon"]) - old_stats["epsilon"]) <= 1e-12
    assert abs(float(new_stats["epsilon"]) - old_stats["epsilon_max"]) <= 1e-12
    assert abs(float(new_stats["disagreement"]) - old_stats["disagreement"]) <= 1e-12
    assert abs(float(new_stats["raw_epsilon"]) - old_stats["raw_epsilon"]) <= 1e-12
    assert int(bool(new_stats["cap_hit"])) == old_stats["cap_hit_calls"]
    assert int(float(new_stats["epsilon"]) >= 0.10) == old_stats["epsilon_ge_010_calls"]
    assert int(float(new_stats["epsilon"]) >= 0.25) == old_stats["epsilon_ge_025_calls"]


def test_uncertainty_generalization_matches_accepted_oracle_below_cap() -> None:
    _assert_fixture(
        [
            (1.0, -0.2, 0.5, 0.1, 0.2, -0.3),
            (0.8, 0.1, 0.4, 0.2, -0.1, 0.0),
            (1.2, -0.4, 0.6, 0.0, 0.3, 0.1),
            (0.9, 0.0, 0.45, 0.05, 0.1, -0.2),
        ]
    )


def test_uncertainty_generalization_matches_accepted_oracle_when_cap_hits() -> None:
    _assert_fixture(
        [
            (10.0, -1.0, -1.0, -1.0, -1.0, -1.0),
            (-1.0, 10.0, -1.0, -1.0, -1.0, -1.0),
            (-1.0, -1.0, 10.0, -1.0, -1.0, -1.0),
            (-1.0, -1.0, -1.0, -1.0, -1.0, 10.0),
        ]
    )
