from __future__ import annotations

import argparse
from pathlib import Path

import r7_5_arch_reset_v1plus_phase2b10_private_public_chance_decomposition as p
from spincore.r7_5_action_scenarios import action_scenario_cycle
from spincore.r7_5_representation_v3 import H2_FINAL
from spincore.r7_5_representation_v3_stage_contract import TRAINING_SEEDS, validate_phase2_v3_contract
from spincore.solver import DealSnapshot, SolverLibrary
from spincore.solver_v3 import neural_bytes_v3


def _metric(tv: float) -> dict:
    return {
        "regret_matching_policy_tv": {"mean": float(tv)},
        "target_mean_abs_diff": {"mean": 0.0},
        "legal_sign_disagreement_fraction": {"mean": 0.0},
        "dominant_legal_action_mismatch_rate": 0.0,
    }


def _decision_fixture(traversal: float, private: float, public: float, combined: float):
    pooled = {
        "TRAVERSAL_ONLY": _metric(traversal),
        "PRIVATE_ONLY": _metric(private),
        "PUBLIC_ONLY": _metric(public),
        "COMBINED": _metric(combined),
    }
    by_seed = {}
    for seed in TRAINING_SEEDS:
        by_seed[str(int(seed))] = {arm: dict(row) for arm, row in pooled.items()}
    return by_seed, pooled


def test_resampling_contract() -> None:
    base = DealSnapshot(
        holes=((0, 1), (2, 3), (4, 5)),
        board=(6, 7, 8, 9, 10),
        visible_board_count=0,
    )
    for arm in p.ARMS:
        first = p._resample_deal(base, 0, arm, 3, 2, 1)
        second = p._resample_deal(base, 0, arm, 3, 2, 1)
        assert first == second
        cards = [x for row in first.holes for x in row] + list(first.board)
        assert len(cards) == 11 and len(set(cards)) == 11
        assert first.holes[0] == base.holes[0]
        if arm in ("TRAVERSAL_ONLY", "PRIVATE_ONLY"):
            assert first.board == base.board
        if arm in ("TRAVERSAL_ONLY", "PUBLIC_ONLY"):
            assert first.holes == base.holes
    private_variants = {p._resample_deal(base, 0, "PRIVATE_ONLY", r, 2, 1).holes for r in range(8)}
    public_variants = {p._resample_deal(base, 0, "PUBLIC_ONLY", r, 2, 1).board for r in range(8)}
    assert len(private_variants) > 1
    assert len(public_variants) > 1


def test_pair_metrics_zero() -> None:
    row = [0.0] * 10
    row[1] = 0.1
    row[2] = -0.1
    metrics = p._pair_metrics([row[:] for _ in range(p.REPLICATES)], [0, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    assert len(metrics) == 4
    assert all(m["target_mean_abs_diff"] == 0.0 for m in metrics)
    assert all(m["legal_sign_disagreement_fraction"] == 0.0 for m in metrics)
    assert all(m["regret_matching_policy_tv"] == 0.0 for m in metrics)
    assert all(m["dominant_legal_action_mismatch"] == 0 for m in metrics)


def test_classification_rules() -> None:
    by_seed, pooled = _decision_fixture(0.05, 0.15, 0.40, 0.43)
    assert p._decision(by_seed, pooled)["classification"] == "PUBLIC_BOARD_CHANCE_DOMINANT"
    by_seed, pooled = _decision_fixture(0.05, 0.40, 0.15, 0.43)
    assert p._decision(by_seed, pooled)["classification"] == "PRIVATE_HOLE_CHANCE_DOMINANT"
    by_seed, pooled = _decision_fixture(0.05, 0.27, 0.30, 0.42)
    assert p._decision(by_seed, pooled)["classification"] == "MIXED_PRIVATE_PUBLIC_CHANCE"
    by_seed, pooled = _decision_fixture(0.05, 0.10, 0.11, 0.13)
    assert p._decision(by_seed, pooled)["classification"] == "CHANCE_COMPONENT_DECOMPOSITION_UNRESOLVED"


def test_solver_roundtrip(repo_root: Path, solver_path: Path) -> None:
    solver = SolverLibrary(solver_path)
    assert solver.explicit_deal_available
    episode = action_scenario_cycle(p.DOMAIN)[0]
    contract = validate_phase2_v3_contract(
        repo_root,
        representation=H2_FINAL,
        domain=p.DOMAIN,
        training_seed=int(TRAINING_SEEDS[0]),
    )
    root = solver.create(episode, 0x2B10F00D12345678)
    try:
        snapshot = root.deal_snapshot()
        assert snapshot.visible_board_count == 0
        original_obs, original_actor, original_legal, original_mask = p.b1._root_identity(root, contract["action_spec"])
        recreated = solver.create_with_deal(episode, snapshot.holes, snapshot.board)
        try:
            assert recreated.deal_snapshot() == snapshot
            recreated_obs, actor, legal, mask = p.b1._root_identity(recreated, contract["action_spec"])
            assert recreated_obs == original_obs
            assert actor == original_actor and legal == original_legal and mask == original_mask
            action = original_legal[0]
            payload_v2 = root.neural_bytes_v2()
            street = int(payload_v2[112])
            active_mask = int(contract["action_spec"].active_mask(street))
            a = root.child_universal(active_mask, action)
            b = recreated.child_universal(active_mask, action)
            try:
                assert a.terminal == b.terminal
                if a.terminal:
                    assert a.terminal_chip_delta() == b.terminal_chip_delta()
                else:
                    assert neural_bytes_v3(a) == neural_bytes_v3(b)
            finally:
                a.close(); b.close()
        finally:
            recreated.close()

        for arm in ("PRIVATE_ONLY", "PUBLIC_ONLY", "COMBINED"):
            variant = p._resample_deal(snapshot, original_actor, arm, 0, 0, 0)
            state = solver.create_with_deal(episode, variant.holes, variant.board)
            try:
                obs, actor, legal, mask = p.b1._root_identity(state, contract["action_spec"])
                assert obs == original_obs
                assert actor == original_actor and legal == original_legal and mask == original_mask
            finally:
                state.close()
    finally:
        root.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--solver", type=Path)
    args = parser.parse_args()
    test_resampling_contract()
    test_pair_metrics_zero()
    test_classification_rules()
    if (args.repo_root is None) != (args.solver is None):
        raise SystemExit("--repo-root and --solver must be supplied together")
    if args.repo_root is not None:
        test_solver_roundtrip(args.repo_root.resolve(), args.solver.resolve())
    print("R7.5 architecture-reset Phase2B10 private/public chance tests PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
