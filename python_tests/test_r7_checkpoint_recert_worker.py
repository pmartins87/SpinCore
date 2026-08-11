from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from spincore.solver import SolverLibrary


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

spec = importlib.util.spec_from_file_location(
    "r7_3_frozen_candidate_checkpoint_worker_test",
    TOOLS / "r7_3_frozen_candidate_checkpoint_worker.py",
)
assert spec and spec.loader
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def _args(iterations: int):
    return SimpleNamespace(
        device="cpu",
        lr=1e-3,
        reservoir_capacity=5000,
        advantage_chunk_steps=8,
        advantage_max_steps_per_iteration=8,
        advantage_fit_target=10.0,
        policy_chunk_steps=8,
        policy_max_steps=8,
        policy_fit_target=10.0,
        batch_size=16,
        audit_size=32,
        cross_seed_per_seed=16,
        exact_opponent_levels=1,
        roots_per_iteration=1,
        iterations=iterations,
    )


def test_uncertainty_checkpoint_branch_is_exact(tmp_path):
    freeze = {
        "behavior_kind": "uncertainty_damping",
        "behavior_semantic_id": "TEST_UNCERTAINTY",
        "ensemble_size": 2,
        "params": {"epsilon_scale": 1.0, "epsilon_cap": 0.5},
    }
    solver = SolverLibrary(ROOT / "build" / "libspincore_solver_c.so")
    report, _continuous, _restored = worker._run_seed(
        seed=12345,
        freeze=freeze,
        solver=solver,
        args=_args(2),
        split_iteration=1,
        checkpoint_dir=tmp_path,
    )
    assert report["all_exact"] is True
    assert all(report["checks"].values())


def test_temporal_checkpoint_preserves_previous_generation_exactly(tmp_path):
    freeze = {
        "behavior_kind": "temporal_blend",
        "behavior_semantic_id": "TEST_TEMPORAL",
        "ensemble_size": 2,
        "params": {"current_policy_weight": 0.5},
    }
    solver = SolverLibrary(ROOT / "build" / "libspincore_solver_c.so")
    report, _continuous, _restored = worker._run_seed(
        seed=54321,
        freeze=freeze,
        solver=solver,
        args=_args(3),
        split_iteration=2,
        checkpoint_dir=tmp_path,
    )
    assert report["all_exact"] is True
    assert report["continuous_finish"]["previous_member_count"] == 2
    assert report["restored_finish"]["previous_member_count"] == 2
    assert all(report["checks"].values())
