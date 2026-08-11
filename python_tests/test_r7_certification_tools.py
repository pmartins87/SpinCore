from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


freeze = _load("freeze_r7_3_candidate_semantics_test", "tools/freeze_r7_3_candidate_semantics.py")
fresh = _load("run_r7_3_frozen_candidate_fresh_repro_test", "tools/run_r7_3_frozen_candidate_fresh_repro.py")
checkpoint_orchestrator = _load(
    "run_r7_3_frozen_candidate_checkpoint_recert_test",
    "tools/run_r7_3_frozen_candidate_checkpoint_recert.py",
)
acceptance640 = _load(
    "run_r7_3_frozen_candidate_640_acceptance_test",
    "tools/run_r7_3_frozen_candidate_640_acceptance.py",
)
proposal = _load("propose_r7_3_winner_test", "tools/propose_r7_3_winner.py")
materialize = _load("materialize_r7_3_winner_selection_test", "tools/materialize_r7_3_winner_selection.py")


def test_freeze_accepts_direct_and_matrix_bound_uncertainty_parameters():
    direct = "--epsilon-scale 1.25 --epsilon-cap 0.50"
    freeze._require_bound_numeric(direct, option="--epsilon-scale", variable="scale", value=1.25)
    freeze._require_bound_numeric(direct, option="--epsilon-cap", variable="cap", value=0.50)

    matrix = """
      - label: s175
        scale: '1.75'
        cap: '0.50'
      python runner.py --epsilon-scale "$scale" --epsilon-cap "$cap"
    """
    freeze._require_bound_numeric(matrix, option="--epsilon-scale", variable="scale", value=1.75)
    freeze._require_bound_numeric(matrix, option="--epsilon-cap", variable="cap", value=0.50)


def test_freeze_rejects_unbound_selected_parameter():
    workflow = "scale: '1.25'\npython runner.py --epsilon-scale 1.00"
    with pytest.raises(SystemExit):
        freeze._require_bound_numeric(workflow, option="--epsilon-scale", variable="scale", value=1.25)


def test_freeze_evidence_hash_is_byte_exact():
    payload = b'{"r7_3_pass":true}\n'
    assert freeze._sha256_bytes(payload) == hashlib.sha256(payload).hexdigest()
    mutated = b'{"r7_3_pass":true} \n'
    assert freeze._sha256_bytes(mutated) != freeze._sha256_bytes(payload)


def test_fresh_repro_compare_ignores_only_clock_fields_and_uses_1e9_tolerance():
    a = {
        "generated_at_unix": 1.0,
        "duration_seconds": 100.0,
        "metric": 0.25,
        "nested": {"count": 7, "flag": True, "text": "same"},
    }
    b = {
        "generated_at_unix": 9.0,
        "duration_seconds": 200.0,
        "metric": 0.25 + 5e-10,
        "nested": {"count": 7, "flag": True, "text": "same"},
    }
    assert fresh._compare(a, b) == []

    b["metric"] = 0.25 + 2e-9
    diffs = fresh._compare(a, b)
    assert len(diffs) == 1
    assert diffs[0]["path"] == "$.metric"
    assert diffs[0]["kind"] == "NUMBER"


def test_certification_contract_keeps_frozen_r7_3_thresholds_and_run_identity():
    assert freeze.FROZEN_GATES == {
        "advantage_weighted_nrmse_max": 0.75,
        "policy_weighted_mean_tv_max": 0.12,
        "cross_seed_mean_tv_max": 0.15,
        "cross_seed_p95_tv_max": 0.35,
    }
    assert freeze.ALGORITHM_SEEDS == (20260829, 20260807)
    assert freeze.EXECUTION_CONTRACT["iterations"] == 5
    assert freeze.EXECUTION_CONTRACT["roots_per_iteration"] == 64
    assert freeze.EXECUTION_CONTRACT["exact_opponent_levels"] == 2
    assert freeze.EXECUTION_CONTRACT["reservoir_capacity"] == 100000
    assert freeze.EXECUTION_CONTRACT["lr"] == 0.001
    assert freeze.EXECUTION_CONTRACT["device"] == "cpu"
    assert freeze.FREEZE_SCHEMA == "SPINCORE_R7_3_CANDIDATE_SEMANTIC_FREEZE_V1"
    assert fresh.REPORT_SCHEMA == "SPINCORE_R7_3_FROZEN_CANDIDATE_FRESH_REPRO_V1"
    assert checkpoint_orchestrator.RECERT_SCHEMA == "SPINCORE_R7_3_CANDIDATE_CHECKPOINT_RECERT_V1"
    assert acceptance640.REPORT_SCHEMA == "SPINCORE_R7_3_FROZEN_CANDIDATE_640_ACCEPTANCE_V1"


def test_winner_proposal_and_selection_schemas_preserve_deliberate_selection():
    assert proposal.SCHEMA == "SPINCORE_R7_3_WINNER_PROPOSAL_V1"
    assert proposal.PROVENANCE_SCHEMA == "SPINCORE_R7_3_ACTIVE_CANDIDATE_PROVENANCE_V1"
    assert proposal.SEEDS == [20260829, 20260807]
    assert proposal.MEAN_GATE == 0.15
    assert proposal.P95_GATE == 0.35
    assert materialize.PROPOSAL_SCHEMA == proposal.SCHEMA
    assert materialize.SELECTION_SCHEMA == "SPINCORE_R7_3_WINNER_SELECTION_V1"
