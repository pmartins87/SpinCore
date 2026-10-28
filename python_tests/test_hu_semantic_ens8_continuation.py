#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "tools" / "run_hu_semantic_ens8_continuation_10105_plus469.py").read_text(encoding="utf-8")
PREREG = json.loads((ROOT / "contracts" / "hu_semantic_ens8_continuation_10105_plus469_v1.json").read_text(encoding="utf-8"))


def test_hu_semantic_architecture_and_budget_are_frozen():
    a = PREREG["semantic_architecture"]
    t = PREREG["training"]
    assert a["semantic_dim"] == 30
    assert a["ensemble_size"] == 8
    assert a["member_fit_steps"] == 1600
    assert t["domain"] == "TRUE_HEADS_UP"
    assert t["roots_per_update"] == 273
    assert t["updates_in_block"] == 469
    assert t["new_hu_roots"] == 128037
    assert "MEMBER_STEPS = 1600" in SOURCE
    assert "ROOTS_PER_UPDATE = 273" in SOURCE
    assert "UPDATES_IN_BLOCK = 469" in SOURCE


def test_migration_gate_is_mechanical_and_fail_closed():
    gate = PREREG["migration_gate"]
    assert gate["parity_sample_count"] == 20000
    assert gate["max_raw_abs_error"] == 0.000002
    assert gate["max_policy_abs_error"] == 0.000002
    assert gate["max_argmax_mismatches"] == 0
    assert "expand_v1_state_with_zero_semantics" in SOURCE
    assert "HU semantic zero-column migration parity failed" in SOURCE
    assert "HU_SEMANTIC_MIGRATION_PARITY_PASS" in SOURCE


def test_hu_only_direct_teacher_path_has_no_average_policy_or_3h_training():
    assert '"three_handed_training_performed": False' in SOURCE
    assert '"average_policy_training_performed": False' in SOURCE
    assert '"strategy_reservoir_collection_performed": False' in SOURCE
    assert "force_domain=DOMAIN_HU" in SOURCE
    assert "SemanticParallelRootExecutor" in SOURCE
    assert "_collect_policy_from_episodes" not in SOURCE
    assert "DOMAIN_3H" not in SOURCE


def test_semantic_online_fit_uses_validated_parallel_fresh1600_path():
    assert "semantic_fit.PackedSemanticAdvantageReservoir.build" in SOURCE
    assert "semantic_fit.make_fit_contract" in SOURCE
    assert "semantic_fit.ParallelSemanticEnsembleFitter" in SOURCE
    assert "fit_semantic_ensemble(fitter)" in SOURCE
    assert "protected_diagnostic_count" in SOURCE


def test_source_identities_are_immutable():
    source = PREREG["source"]
    assert source["checkpoint_sha256"] == "f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
    assert source["hu_ensemble_sha256"] == "8d11cb6bce172e24e903ced650c7a7d82aeb2b251c71c3cfc28e37d873ddb62d"
    assert "frozen HU source artifacts changed during semantic continuation" in SOURCE
