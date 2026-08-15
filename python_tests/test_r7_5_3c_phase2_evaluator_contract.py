from __future__ import annotations

import json
import subprocess
from pathlib import Path

from spincore.r7_5_representation_v3_referee_rng import canonical_key


def _read(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_phase2_evaluator_is_frozen_before_evaluation_outputs() -> None:
    freeze = _read("validation/R7_5_3C_PHASE2_EVALUATOR_IMPLEMENTATION_FREEZE_20260815.json")
    assert freeze["schema"] == "SPINCORE_R7_5_3C_PHASE2_EVALUATOR_IMPLEMENTATION_FREEZE_V1"
    assert freeze["status"] == "FROZEN_BEFORE_PHASE2_EVALUATION_OUTPUTS"
    assert freeze["phase2_evaluation_output_seen_before_freeze"] is False
    assert freeze["training_run"] == {
        "run_id": 31865993350,
        "execution_sha": "9b0ccc207135c3adaec76ea87de8ec21f7415957",
        "required_final_cells": 8,
        "must_complete_successfully_before_evaluation": True,
    }
    assert freeze["heldout_run"]["run_id"] == 31866538984
    assert freeze["heldout_run"]["execution_sha"] == "dfe5f83742495a457e92b29f97db5d3b631bca22"
    assert freeze["evaluation_matrix"]["heldout_cells"] == 16
    assert freeze["evaluation_matrix"]["common_reference_cells"] == 16
    assert freeze["evaluation_matrix"]["pairwise_cells"] == 16
    assert freeze["aggregation_contract"]["bootstrap_replicates"] == 2000
    assert freeze["aggregation_contract"]["material_floor_icm"] == 0.001
    assert freeze["aggregation_contract"]["domain_conflicts_must_block_instead_of_being_averaged_away"] is True
    assert freeze["aggregation_contract"]["common_reference_selects_winner"] is False
    assert freeze["aggregation_contract"]["local_deviation_called_exact_exploitability"] is False
    assert freeze["production_training_authorized"] is False
    assert freeze["ready_for_tables"] is False


def test_phase2_evaluator_frozen_blob_hashes_match_repository() -> None:
    freeze = _read("validation/R7_5_3C_PHASE2_EVALUATOR_IMPLEMENTATION_FREEZE_20260815.json")
    for table in ("new_git_blob_freeze", "inherited_evaluator_blob_freeze"):
        for path, expected in freeze[table].items():
            actual = subprocess.check_output(["git", "hash-object", path], text=True).strip()
            assert actual == expected, (path, actual, expected)


def test_refv1_namespace_adjudication_is_constant_and_candidate_independent() -> None:
    adj = _read("validation/R7_5_3C_PHASE2_REFEREE_RNG_NAMESPACE_ADJUDICATION_20260815.json")
    assert adj["schema"] == "SPINCORE_R7_5_3C_PHASE2_REFEREE_RNG_NAMESPACE_ADJUDICATION_V1"
    assert adj["status"] == "ADJUDICATED_BEFORE_PHASE2_EVALUATION_OUTPUTS"
    binding = adj["binding_interpretation"]
    assert binding["canonical_prefix"] == "SpinCore|R7.5.3C|PHASE2|REFV1"
    assert binding["candidate_identity_in_rng_key"] is False
    assert binding["representation_identity_in_rng_key"] is False
    assert binding["training_seed_identity_in_rng_key"] is False
    key = canonical_key("heldout", "TRUE_HEADS_UP", 2029384436, 17, "deck")
    assert key.startswith("SpinCore|R7.5.3C|PHASE2|REFV1|heldout|")
    assert "H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL" not in key
    assert "H3_HYBRID_EXACT_SEMANTIC_FINAL" not in key
    assert "1342191342" not in key
    assert "1801739323" not in key
    assert adj["selection_rule_changed"] is False
    assert adj["phase2_evaluation_output_seen_before_adjudication"] is False


def test_phase2_evaluator_workflow_is_bound_to_frozen_runs_and_tools() -> None:
    workflow = Path(".github/workflows/r7_5_3c_phase2_evaluation.yml").read_text(encoding="utf-8")
    for literal in (
        "31865993350",
        "9b0ccc207135c3adaec76ea87de8ec21f7415957",
        "31866538984",
        "dfe5f83742495a457e92b29f97db5d3b631bca22",
        "tools/r7_5_3c_phase2_extract_final_policies.py",
        "tools/r7_5_3c_phase2_eval_worker.py",
        "tools/r7_5_3c_phase2_eval_aggregate.py",
        "r7-5-3c-p2-strategic-result-",
    ):
        assert literal in workflow
    assert "phase2_evaluation_output_seen_before_freeze" in workflow
    assert "integrity_complete" in workflow
    assert "training_quality_pass is evidence" in workflow
