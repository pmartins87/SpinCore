from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import r7_5_4_strategic_preflight as preflight


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _complete_tree(root: Path) -> None:
    v = root / "validation"
    v.mkdir(parents=True, exist_ok=True)
    (v / preflight.HERITAGE_AUDIT).write_text("full heritage audit\n", encoding="utf-8")
    readable_sources = [
        {"source": f"source_{i}", "full_read_status": "text_full_read"}
        for i in range(17)
    ]
    _write(
        v / preflight.HERITAGE_MANIFEST,
        {
            "schema": preflight.HERITAGE_SCHEMA,
            "ready_for_tables": False,
            "prior_attempt_archive": {
                "readable_sources": readable_sources,
                "archive_comparison": {
                    "shared_entries": 27,
                    "shared_byte_identical": 27,
                    "only_in_superset": [
                        "hardcoded/Crusher Framework 5.txt",
                        "solver v2/184Flops.json",
                    ],
                },
            },
        },
    )
    _write(
        v / preflight.HERITAGE_LEDGER,
        {
            "schema": preflight.HERITAGE_LEDGER_SCHEMA,
            "entries": [
                {
                    "source": row["source"],
                    "reviewed_full": True,
                    "best_of": ["best useful detail"],
                    "preserve": ["preserve objective lesson"],
                    "destination": "current or future roadmap destination",
                    "status": "TEST_FIXTURE_DISPOSITION",
                    "current_evidence": ["test evidence"],
                    "never_inherit_as_truth": ["old strategy output"],
                }
                for row in readable_sources
            ],
            "strategic_output": False,
            "production_training_authorized": False,
            "ready_for_tables": False,
        },
    )
    _write(
        v / preflight.REACHABILITY_CONTRACT,
        {
            "schema": preflight.REACHABILITY_SCHEMA,
            "required_invariants": {
                key: True for key in preflight.REQUIRED_REACHABILITY_INVARIANTS
            },
            "production_training_authorized": False,
            "ready_for_tables": False,
        },
    )
    _write(
        v / preflight.REACHABILITY_AUDIT,
        {
            "schema": preflight.REACHABILITY_AUDIT_SCHEMA,
            "reachability_gate_pass": True,
            "claims": {
                key: True for key in preflight.REQUIRED_REACHABILITY_CLAIMS
            },
            "totals": {
                "trajectories": 60,
                "decisions": 263,
                "v1_legal_mask_checks": 263,
                "v2_legal_mask_checks": 263,
                "legacy_illegal_rejections": 263,
                "universal_illegal_rejections": 202,
                "history_monotonic_checks": 263,
                "terminal_chip_conservation_checks": 60,
                "terminal_icm_conservation_checks": 60,
            },
            "production_training_authorized": False,
            "ready_for_tables": False,
        },
    )
    _write(
        v / "R7_5_3_REPRESENTATION_ABLATION_RESULT.json",
        {
            "schema": preflight.REP_SCHEMA,
            "r7_5_3_representation_ablation_pass": True,
            "selected_candidate": "C0_V1_FROZEN_CONTROL",
            "production_training_authorized": False,
            "ready_for_tables": False,
        },
    )
    _write(
        v / preflight.TRAINABILITY_CONTRACT,
        {
            "schema": preflight.TRAINABILITY_SCHEMA,
            "hard_cap": {"wall_clock_days": 90.0},
            "planning_reserve": {
                "multiplier": 1.20,
                "implied_nominal_budget_days": 75.0,
            },
            "representation_state_at_freeze": {
                "selected_candidate": "C0_V1_FROZEN_CONTROL",
                "serialized_observation_bytes": 126,
                "model_parameter_count": 152438,
            },
            "physical_measurement_contract": {
                "timing_cost_basis": "MATURE_OR_WORST_CASE_CERTIFIED_V1",
                "non_iteration_scope": "ALL_FROZEN_NON_ITERATION_TRAINING_AND_FINAL_FREEZE_WORK_V1",
            },
            "projection_contract": {
                "schema": preflight.TRAINABILITY_PROJECTION_SCHEMA,
                "required_domains": ["TRUE_HEADS_UP", "THREE_HANDED"],
                "all_selected_profiles_required": True,
                "all_required_algorithm_seed_streams_required": True,
                "all_frozen_iterations_required": True,
                "all_non_iteration_training_and_freeze_work_required": True,
            },
            "current_trainability_status": "NOT_MEASURED_PHYSICALLY / NOT_PASS",
            "production_training_authorized": False,
            "ready_for_tables": False,
        },
    )
    _write(
        v / "R7_5_4_ACTION_STRUCTURAL_AUDIT.json",
        {
            "schema": preflight.STRUCT_SCHEMA,
            "structural_gate_pass": True,
            "production_training_authorized": False,
            "ready_for_tables": False,
        },
    )
    _write(
        v / "R7_5_4_UNCERTAINTY_EQUIVALENCE.json",
        {
            "schema": preflight.UNCERTAINTY_SCHEMA,
            "uncertainty_equivalence_pass": True,
            "maximum_abs_difference": 0.0,
            "production_training_authorized": False,
            "ready_for_tables": False,
        },
    )
    for name, schema in (
        ("R7_5_4_ACTION_ABSTRACTION_ABLATION_PRECOMMIT.json", preflight.PRECOMMIT_V1_SCHEMA),
        ("R7_5_4_ACTION_ABSTRACTION_ABLATION_PRECOMMIT_V2.json", preflight.PRECOMMIT_V2_SCHEMA),
        ("R7_5_4_ACTION_ABSTRACTION_ABLATION_PRECOMMIT_V3.json", preflight.PRECOMMIT_V3_SCHEMA),
        ("R7_5_4_TRAINING_IMPLEMENTATION_FREEZE.json", preflight.TRAINING_FREEZE_SCHEMA),
    ):
        _write(v / name, {"schema": schema, "ready_for_tables": False})


def test_preflight_allows_only_initial_160_when_all_durable_dependencies_pass(tmp_path: Path) -> None:
    _complete_tree(tmp_path)
    result = preflight.evaluate(tmp_path, phase="R7_5_4A_POSTFLOP", root_level=160)
    assert result["schema"] == "SPINCORE_R7_5_4_STRATEGIC_PREFLIGHT_V5"
    assert result["ready_to_start"] is True
    assert result["selected_representation"] == "C0_V1_FROZEN_CONTROL"
    assert result["physical_trainability_pass_required_now"] is False
    assert result["physical_trainability_gate_required_before_r8_official_training"] is True
    assert result["production_training_authorized"] is False
    assert result["ready_for_tables"] is False

    higher = preflight.evaluate(tmp_path, phase="R7_5_4A_POSTFLOP", root_level=320)
    assert higher["ready_to_start"] is False
    assert higher["checks"]["initial_postflop_level"]["pass"] is False


def test_preflight_fails_closed_when_trainability_contract_is_missing(tmp_path: Path) -> None:
    _complete_tree(tmp_path)
    (tmp_path / "validation" / preflight.TRAINABILITY_CONTRACT).unlink()
    result = preflight.evaluate(tmp_path, phase="R7_5_4A_POSTFLOP", root_level=160)
    assert result["ready_to_start"] is False
    assert result["checks"]["trainability_contract_available"]["pass"] is False


def test_preflight_fails_closed_when_trainability_budget_drifts(tmp_path: Path) -> None:
    _complete_tree(tmp_path)
    path = tmp_path / "validation" / preflight.TRAINABILITY_CONTRACT
    payload = json.loads(path.read_text())
    payload["hard_cap"]["wall_clock_days"] = 120.0
    _write(path, payload)
    result = preflight.evaluate(tmp_path, phase="R7_5_4A_POSTFLOP", root_level=160)
    assert result["ready_to_start"] is False
    assert result["checks"]["trainability_budget_frozen"]["pass"] is False


def test_preflight_rejects_premature_physical_trainability_pass_contamination(tmp_path: Path) -> None:
    _complete_tree(tmp_path)
    path = tmp_path / "validation" / preflight.TRAINABILITY_CONTRACT
    payload = json.loads(path.read_text())
    payload["current_trainability_status"] = "PASS"
    payload["production_training_authorized"] = True
    _write(path, payload)
    result = preflight.evaluate(tmp_path, phase="R7_5_4A_POSTFLOP", root_level=160)
    assert result["ready_to_start"] is False
    assert result["checks"]["trainability_pre_physical_status"]["pass"] is False


def test_preflight_fails_closed_when_uncertainty_evidence_is_missing(tmp_path: Path) -> None:
    _complete_tree(tmp_path)
    (tmp_path / "validation" / "R7_5_4_UNCERTAINTY_EQUIVALENCE.json").unlink()
    result = preflight.evaluate(tmp_path, phase="R7_5_4A_POSTFLOP", root_level=160)
    assert result["ready_to_start"] is False
    assert result["checks"]["uncertainty_evidence_available"]["pass"] is False


def test_preflight_fails_closed_on_ready_for_tables_contamination(tmp_path: Path) -> None:
    _complete_tree(tmp_path)
    path = tmp_path / "validation" / "R7_5_4_ACTION_STRUCTURAL_AUDIT.json"
    payload = json.loads(path.read_text())
    payload["ready_for_tables"] = True
    _write(path, payload)
    result = preflight.evaluate(tmp_path, phase="R7_5_4A_POSTFLOP", root_level=160)
    assert result["ready_to_start"] is False
    assert result["checks"]["structural_not_table_authority"]["pass"] is False


def test_preflight_fails_closed_when_full_heritage_manifest_is_missing(tmp_path: Path) -> None:
    _complete_tree(tmp_path)
    (tmp_path / "validation" / preflight.HERITAGE_MANIFEST).unlink()
    result = preflight.evaluate(tmp_path, phase="R7_5_4A_POSTFLOP", root_level=160)
    assert result["ready_to_start"] is False
    assert result["checks"]["heritage_manifest_available"]["pass"] is False
    assert result["checks"]["heritage_ledger_complete"]["pass"] is False


def test_preflight_fails_closed_when_heritage_ledger_is_missing(tmp_path: Path) -> None:
    _complete_tree(tmp_path)
    (tmp_path / "validation" / preflight.HERITAGE_LEDGER).unlink()
    result = preflight.evaluate(tmp_path, phase="R7_5_4A_POSTFLOP", root_level=160)
    assert result["ready_to_start"] is False
    assert result["checks"]["heritage_ledger_available"]["pass"] is False


def test_preflight_fails_closed_when_heritage_ledger_omits_one_readable_source(tmp_path: Path) -> None:
    _complete_tree(tmp_path)
    path = tmp_path / "validation" / preflight.HERITAGE_LEDGER
    payload = json.loads(path.read_text())
    payload["entries"].pop()
    _write(path, payload)
    result = preflight.evaluate(tmp_path, phase="R7_5_4A_POSTFLOP", root_level=160)
    assert result["ready_to_start"] is False
    assert result["checks"]["heritage_ledger_complete"]["pass"] is False


def test_preflight_fails_closed_when_reachability_invariant_is_false(tmp_path: Path) -> None:
    _complete_tree(tmp_path)
    path = tmp_path / "validation" / preflight.REACHABILITY_CONTRACT
    payload = json.loads(path.read_text())
    payload["required_invariants"]["folded_player_never_acts_again"] = False
    _write(path, payload)
    result = preflight.evaluate(tmp_path, phase="R7_5_4A_POSTFLOP", root_level=160)
    assert result["ready_to_start"] is False
    assert result["checks"]["reachable_state_contract"]["pass"] is False
    assert "folded_player_never_acts_again" in result["checks"]["reachable_state_contract"]["detail"]


def test_preflight_fails_closed_when_reachability_audit_is_missing(tmp_path: Path) -> None:
    _complete_tree(tmp_path)
    (tmp_path / "validation" / preflight.REACHABILITY_AUDIT).unlink()
    result = preflight.evaluate(tmp_path, phase="R7_5_4A_POSTFLOP", root_level=160)
    assert result["ready_to_start"] is False
    assert result["checks"]["reachability_audit_available"]["pass"] is False


def test_preflight_fails_closed_when_reachability_claim_is_false(tmp_path: Path) -> None:
    _complete_tree(tmp_path)
    path = tmp_path / "validation" / preflight.REACHABILITY_AUDIT
    payload = json.loads(path.read_text())
    payload["claims"]["observations_only_from_engine_reached_states"] = False
    _write(path, payload)
    result = preflight.evaluate(tmp_path, phase="R7_5_4A_POSTFLOP", root_level=160)
    assert result["ready_to_start"] is False
    assert result["checks"]["reachability_audit_gate"]["pass"] is False
    assert "observations_only_from_engine_reached_states" in result["checks"]["reachability_audit_gate"]["detail"]


def test_preflight_fails_closed_when_heritage_archive_is_not_byte_equivalent(tmp_path: Path) -> None:
    _complete_tree(tmp_path)
    path = tmp_path / "validation" / preflight.HERITAGE_MANIFEST
    payload = json.loads(path.read_text())
    payload["prior_attempt_archive"]["archive_comparison"]["shared_byte_identical"] = 26
    _write(path, payload)
    result = preflight.evaluate(tmp_path, phase="R7_5_4A_POSTFLOP", root_level=160)
    assert result["ready_to_start"] is False
    assert result["checks"]["heritage_archive_equivalence"]["pass"] is False
