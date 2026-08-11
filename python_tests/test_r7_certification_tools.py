from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
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
evidence_loader = _load("r7_3_certification_evidence_test", "tools/r7_3_certification_evidence.py")
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


def test_frozen_source_thread_contract_rejects_explicit_overrides():
    freeze._require_thread_environment_contract("env:\n  PYTHONPATH: x\n")
    for key in freeze.THREAD_ENV_KEYS:
        with pytest.raises(SystemExit):
            freeze._require_thread_environment_contract(f"env:\n  {key}: '2'\n")


def test_source_execution_environment_never_injects_thread_counts():
    frozen = {"thread_environment_contract": freeze.THREAD_ENV_CONTRACT}
    base = {"PATH": "/bin", "PYTHONHASHSEED": "0"}
    got = fresh._source_execution_env(frozen, base)
    assert got == base
    assert got is not base
    for key in freeze.THREAD_ENV_KEYS:
        assert key not in got

    inherited = {"PATH": "/bin", "OMP_NUM_THREADS": "7"}
    got = fresh._source_execution_env(frozen, inherited)
    assert got["OMP_NUM_THREADS"] == "7"


def test_source_execution_environment_fails_closed_on_unknown_contract():
    with pytest.raises(ValueError):
        fresh._source_execution_env({"thread_environment_contract": "OTHER"}, {"PATH": "/bin"})


def test_checkpoint_worker_overlay_executes_inside_frozen_worktree(tmp_path):
    helper, worker = checkpoint_orchestrator._overlay_targets(tmp_path)
    assert helper == tmp_path / checkpoint_orchestrator.HELPER_REL
    assert worker == tmp_path / checkpoint_orchestrator.WORKER_REL
    assert worker.parent == tmp_path / "tools"
    assert helper.parent == tmp_path / "python" / "spincore"


def _cert_chain():
    source = "a" * 40
    evidence = "b" * 40
    evidence_sha = "c" * 64
    semantic = "SPINCORE_R7_3_UNCERTAINTY_POLICY_MIXTURE_V1"
    contract = freeze.THREAD_ENV_CONTRACT
    frozen = {
        "schema": freeze.FREEZE_SCHEMA,
        "evidence_r7_3_pass": True,
        "source_head_sha": source,
        "evidence_commit_sha": evidence,
        "evidence_sha256": evidence_sha,
        "behavior_semantic_id": semantic,
        "thread_environment_contract": contract,
    }
    fresh_report = {
        "schema": fresh.REPORT_SCHEMA,
        "fresh_process_reproducible": True,
        "difference_count": 0,
        "source_head_sha": source,
        "original_evidence_commit_sha": evidence,
        "original_evidence_sha256": evidence_sha,
        "behavior_semantic_id": semantic,
        "thread_environment_contract": contract,
        "thread_environment_overrides_injected_by_certifier": False,
    }
    checkpoint = {
        "schema": checkpoint_orchestrator.RECERT_SCHEMA,
        "checkpoint_resume_recertification_pass": True,
        "source_head_sha": source,
        "evidence_commit_sha": evidence,
        "behavior_semantic_id": semantic,
        "thread_environment_contract": contract,
        "thread_environment_overrides_injected_by_certifier": False,
        "algorithm_source_exact_worktree": True,
        "checkpoint_worker_executed_from_frozen_worktree_overlay": True,
        "fresh_process_zero_difference_gate_passed_first": True,
        "validated_fresh_original_evidence_sha256": evidence_sha,
        "validated_fresh_behavior_semantic_id": semantic,
        "acceptance_gate_changed": False,
    }
    return frozen, fresh_report, checkpoint


def test_checkpoint_recert_requires_corrected_zero_difference_fresh_evidence():
    frozen, fresh_report, _checkpoint = _cert_chain()
    checkpoint_orchestrator._validate_fresh_prerequisite(frozen, fresh_report)
    for mutation in (
        lambda x: x.__setitem__("difference_count", 1),
        lambda x: x.__setitem__("thread_environment_overrides_injected_by_certifier", True),
        lambda x: x.pop("thread_environment_contract"),
        lambda x: x.__setitem__("original_evidence_sha256", "d" * 64),
    ):
        bad = copy.deepcopy(fresh_report)
        mutation(bad)
        with pytest.raises(ValueError):
            checkpoint_orchestrator._validate_fresh_prerequisite(frozen, bad)


def test_640_acceptance_requires_full_corrected_certification_provenance():
    frozen, fresh_report, checkpoint = _cert_chain()
    assert acceptance640._validate_certification_chain(frozen, fresh_report, checkpoint) == frozen["source_head_sha"]
    mutations = (
        ("fresh", lambda x: x.__setitem__("difference_count", 2)),
        ("fresh", lambda x: x.__setitem__("thread_environment_overrides_injected_by_certifier", True)),
        ("checkpoint", lambda x: x.__setitem__("checkpoint_worker_executed_from_frozen_worktree_overlay", False)),
        ("checkpoint", lambda x: x.__setitem__("fresh_process_zero_difference_gate_passed_first", False)),
        ("checkpoint", lambda x: x.__setitem__("thread_environment_overrides_injected_by_certifier", True)),
        ("checkpoint", lambda x: x.__setitem__("validated_fresh_original_evidence_sha256", "d" * 64)),
    )
    for target, mutation in mutations:
        f = copy.deepcopy(fresh_report)
        c = copy.deepcopy(checkpoint)
        mutation(f if target == "fresh" else c)
        with pytest.raises(ValueError):
            acceptance640._validate_certification_chain(frozen, f, c)


def _git(cwd: Path, *args: str) -> None:
    subprocess.check_call(["git", *args], cwd=cwd)


def test_certification_evidence_resolver_recovers_valid_history_after_stale_overwrite(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "SpinCore Test")
    evidence_path = repo / "validation" / "cert.json"
    evidence_path.parent.mkdir()

    valid = {"schema": "CORRECTED", "pass": True}
    evidence_path.write_text(json.dumps(valid) + "\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "valid")
    valid_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()

    stale = {"schema": "LEGACY", "pass": True}
    evidence_path.write_text(json.dumps(stale) + "\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "legacy overwrite")

    def validator(obj):
        if obj.get("schema") != "CORRECTED" or obj.get("pass") is not True:
            raise ValueError("not corrected evidence")

    resolved, origin = evidence_loader.resolve_valid_json(
        "validation/cert.json",
        validator=validator,
        repo_root=repo,
    )
    assert resolved == valid
    assert origin["history_fallback_used"] is True
    assert origin["origin"] == "GIT_HISTORY"
    assert origin["commit_sha"] == valid_commit
    assert origin["rejected_newer_versions"]


def test_certification_evidence_resolver_fails_closed_when_no_valid_version(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "SpinCore Test")
    path = repo / "cert.json"
    path.write_text('{"schema":"LEGACY"}\n', encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "legacy only")

    with pytest.raises(ValueError):
        evidence_loader.resolve_valid_json(
            "cert.json",
            validator=lambda obj: (_ for _ in ()).throw(ValueError("reject")),
            repo_root=repo,
        )
