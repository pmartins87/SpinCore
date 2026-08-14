from __future__ import annotations

import argparse
import hashlib
import json
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

SCHEMA = "SPINCORE_R7_5_4_UNCERTAINTY_EQUIVALENCE_AUDIT_V1"
TOLERANCE = 1e-12


class FixedSix(torch.nn.Module):
    def __init__(self, row):
        super().__init__()
        self.register_buffer("row", torch.tensor(row, dtype=torch.float32))

    def forward(self, batch):
        n = int(batch["numeric"].shape[0])
        return self.row.unsqueeze(0).expand(n, -1)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def episode() -> Episode:
    return Episode(1500, True, 0, 10, 20, (0, 750, 750), 1, (0,))


def oracle(rows, solver: SolverLibrary):
    state = solver.create(episode(), 123456)
    try:
        observation = state.neural_bytes()
        legal = state.legal_actions()
        behavior = accepted.UncertaintyDampedPolicyMixture(device="cpu")
        behavior.models = [FixedSix(row) for row in rows]
        policy = tuple(float(value) for value in behavior(state, observation, legal))
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
        return policy, tuple(int(x) for x in legal), stats
    finally:
        state.close()


def run_fixture(name: str, rows, solver: SolverLibrary) -> dict:
    rows_f32 = [
        tuple(float(value) for value in torch.tensor(row, dtype=torch.float32).tolist())
        for row in rows
    ]
    old_policy, legal, old_stats = oracle(rows_f32, solver)
    widened = [row + (0.0, 0.0, 0.0, 0.0) for row in rows_f32]
    new_policy, new_stats = uncertainty_damped_policy_from_advantages(
        widened,
        legal,
        action_count=10,
        epsilon_scale=1.75,
        epsilon_cap=0.5,
    )
    policy_differences = [
        abs(float(new_policy[action]) - float(old_policy[action])) for action in range(6)
    ]
    illegal_tail_max = max(abs(float(value)) for value in new_policy[6:])
    stat_differences = {
        "epsilon": abs(float(new_stats["epsilon"]) - old_stats["epsilon"]),
        "epsilon_max": abs(float(new_stats["epsilon"]) - old_stats["epsilon_max"]),
        "disagreement": abs(float(new_stats["disagreement"]) - old_stats["disagreement"]),
        "raw_epsilon": abs(float(new_stats["raw_epsilon"]) - old_stats["raw_epsilon"]),
    }
    boolean_equal = bool(
        old_stats["calls"] == 1
        and int(bool(new_stats["cap_hit"])) == old_stats["cap_hit_calls"]
        and int(float(new_stats["epsilon"]) >= 0.10) == old_stats["epsilon_ge_010_calls"]
        and int(float(new_stats["epsilon"]) >= 0.25) == old_stats["epsilon_ge_025_calls"]
    )
    maximum_difference = max(
        [*policy_differences, illegal_tail_max, *stat_differences.values()]
    )
    passed = bool(maximum_difference <= TOLERANCE and boolean_equal)
    return {
        "name": name,
        "legal": list(legal),
        "old_policy": list(old_policy),
        "new_policy_first_six": list(new_policy[:6]),
        "new_policy_illegal_tail": list(new_policy[6:]),
        "old_stats": old_stats,
        "new_stats": dict(new_stats),
        "policy_max_abs_difference": max(policy_differences),
        "illegal_tail_max_abs": illegal_tail_max,
        "stat_abs_differences": stat_differences,
        "maximum_abs_difference": maximum_difference,
        "boolean_statistics_equal": boolean_equal,
        "pass": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="R7.5.4 ten-action uncertainty equivalence audit")
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    solver = SolverLibrary(args.lib)
    fixtures = [
        run_fixture(
            "below_cap",
            [
                (1.0, -0.2, 0.5, 0.1, 0.2, -0.3),
                (0.8, 0.1, 0.4, 0.2, -0.1, 0.0),
                (1.2, -0.4, 0.6, 0.0, 0.3, 0.1),
                (0.9, 0.0, 0.45, 0.05, 0.1, -0.2),
            ],
            solver,
        ),
        run_fixture(
            "cap_hit",
            [
                (10.0, -1.0, -1.0, -1.0, -1.0, -1.0),
                (-1.0, 10.0, -1.0, -1.0, -1.0, -1.0),
                (-1.0, -1.0, 10.0, -1.0, -1.0, -1.0),
                (-1.0, -1.0, -1.0, -1.0, -1.0, 10.0),
            ],
            solver,
        ),
    ]
    passed = all(row["pass"] for row in fixtures)
    payload = {
        "schema": SCHEMA,
        "tolerance": TOLERANCE,
        "accepted_source": "tools/run_r7_3_policy_mixture_uncertainty_damping.py",
        "accepted_source_sha256": sha256(ROOT / "tools" / "run_r7_3_policy_mixture_uncertainty_damping.py"),
        "generalized_source": "python/spincore/r7_5_action_uncertainty.py",
        "generalized_source_sha256": sha256(ROOT / "python" / "spincore" / "r7_5_action_uncertainty.py"),
        "fixtures": fixtures,
        "maximum_abs_difference": max(row["maximum_abs_difference"] for row in fixtures),
        "uncertainty_equivalence_pass": bool(passed),
        "strategic_action_output": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
