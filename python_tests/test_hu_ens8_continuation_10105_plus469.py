#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "tools" / "run_hu_ens8_continuation_10105_plus469.py").read_text(encoding="utf-8")
PREREG = json.loads((ROOT / "contracts" / "hu_ens8_continuation_10105_plus469_v1.json").read_text(encoding="utf-8"))


def test_hu_block_root_budget_is_frozen_and_root_balanced():
    t = PREREG["training"]
    assert t["domain"] == "TRUE_HEADS_UP"
    assert t["roots_per_update"] == 273
    assert t["updates_in_block"] == 469
    assert t["new_hu_roots"] == 128037
    assert "ROOTS_PER_UPDATE = 273" in SOURCE
    assert "UPDATES_IN_BLOCK = 469" in SOURCE
    assert "NEW_ROOTS = ROOTS_PER_UPDATE * UPDATES_IN_BLOCK" in SOURCE


def test_hu_uses_existing_validated_fresh400_ens8_semantics():
    t = PREREG["training"]
    assert t["ensemble_size"] == 8
    assert t["member_fit_steps"] == 400
    assert t["average_policy_training"] is False
    assert t["strategy_reservoir_collection"] is False
    assert t["three_handed_training"] is False
    assert "MEMBER_STEPS = 400" in SOURCE
    assert "par.ParallelEnsembleFitter" in SOURCE
    assert "par.install_ensemble_behavior" in SOURCE
    assert "ensemble_model_states=states" in SOURCE


def test_source_identity_is_fail_closed():
    s = PREREG["source"]
    assert s["checkpoint_sha256"] == "f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
    assert s["hu_ensemble_sha256"] == "8d11cb6bce172e24e903ced650c7a7d82aeb2b251c71c3cfc28e37d873ddb62d"
    assert "frozen 10105 checkpoint SHA mismatch" in SOURCE
    assert "frozen 10105 HU ensemble SHA mismatch" in SOURCE
    assert "source artifacts changed during continuation" in SOURCE


def test_no_three_handed_or_average_policy_training_path():
    assert 'domain=DOMAIN_HU' in SOURCE
    assert 'force_domain=DOMAIN_HU' in SOURCE
    assert '"three_handed_training_performed": False' in SOURCE
    assert '"average_policy_training_performed": False' in SOURCE
    assert '"strategy_reservoir_collection_performed": False' in SOURCE
    assert "_collect_policy_from_episodes" not in SOURCE
    assert "DOMAIN_3H" not in SOURCE


def test_resume_and_nonfinite_guards_exist():
    assert "UniformReservoir.from_state_dict" in SOURCE
    assert "HU resume source checkpoint mismatch" in SOURCE
    assert "HU resume roots/update drift" in SOURCE
    assert "non-finite loss" in SOURCE
    assert "non-finite model tensor" in SOURCE
    assert "HU root-count mismatch" in SOURCE
