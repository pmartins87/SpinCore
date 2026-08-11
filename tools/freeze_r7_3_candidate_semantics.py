from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


SELECTION_SCHEMA = "SPINCORE_R7_3_WINNER_SELECTION_V1"
FREEZE_SCHEMA = "SPINCORE_R7_3_CANDIDATE_SEMANTIC_FREEZE_V1"
DECK_FORMULA = "seed*1000003 + global_root*97 + iteration"
FROZEN_GATES = {
    "advantage_weighted_nrmse_max": 0.75,
    "policy_weighted_mean_tv_max": 0.12,
    "cross_seed_mean_tv_max": 0.15,
    "cross_seed_p95_tv_max": 0.35,
}
EXECUTION_CONTRACT = {
    "iterations": 5,
    "roots_per_iteration": 64,
    "advantage_chunk_steps": 256,
    "advantage_max_steps_per_iteration": 4096,
    "advantage_fit_target": 0.50,
    "policy_chunk_steps": 256,
    "policy_max_steps": 16384,
    "policy_fit_target": 0.105,
    "batch_size": 256,
    "audit_size": 512,
    "cross_seed_per_seed": 1024,
    "reservoir_capacity": 100000,
    "exact_opponent_levels": 2,
}

KIND_CONTRACT = {
    "uncertainty_damping": {
        "schema": "SPINCORE_R7_3_POLICY_MIXTURE_UNCERTAINTY_DAMPING_V1",
        "parameter_block": "uncertainty_damping",
        "required_params": ("epsilon_scale", "epsilon_cap"),
        "runner": "tools/run_r7_3_policy_mixture_uncertainty_damping.py",
    },
    "temporal_blend": {
        "schema": "SPINCORE_R7_3_POLICY_MIXTURE_TEMPORAL_BLEND_V1",
        "parameter_block": "temporal_blend",
        "required_params": ("current_policy_weight",),
        "runner": "tools/run_r7_3_policy_mixture_temporal_blend.py",
    },
    "policy_mixture": {
        "schema": "SPINCORE_R7_3_PARTIAL_EXACT_POLICY_MIXTURE_PAIRED_V1",
        "parameter_block": None,
        "required_params": (),
        "runner": "tools/run_r7_3_partial_exact_policy_mixture_paired.py",
    },
}


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _float_equal(a, b, tol=1e-12) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def _object_sha(ref: str, path: str) -> str:
    try:
        return _git("rev-parse", f"{ref}:{path}")
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"source ref {ref!r} does not contain required path {path!r}") from exc


def _show(ref: str, path: str) -> str:
    try:
        return _git("show", f"{ref}:{path}")
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"cannot read {path!r} at source ref {ref!r}") from exc


def _require_workflow_contract(text: str, ensemble_size: int, kind: str, params: dict) -> None:
    required_fragments = [
        f"--ensemble-size {ensemble_size}",
        "--exact-opponent-levels 2",
        "--iterations 5 --roots-per-iteration 64",
        "--advantage-chunk-steps 256 --advantage-max-steps-per-iteration 4096",
        "--advantage-fit-target 0.50",
        "--policy-chunk-steps 256 --policy-max-steps 16384",
        "--policy-fit-target 0.105",
        "--batch-size 256 --audit-size 512",
        "--cross-seed-per-seed 1024 --reservoir-capacity 100000",
    ]
    if kind == "uncertainty_damping":
        required_fragments.extend([
            f"--epsilon-scale {float(params['epsilon_scale'])}",
            f"--epsilon-cap {float(params['epsilon_cap']):.2f}",
        ])
    elif kind == "temporal_blend":
        required_fragments.append(f"--current-weight {float(params['current_policy_weight']):.2f}")
    missing = [fragment for fragment in required_fragments if fragment not in text]
    if missing:
        raise SystemExit(f"source workflow does not encode the frozen execution contract: missing {missing}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Freeze exact semantics of a gate-clearing R7.3 candidate")
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("validation/R7_3_CANDIDATE_SEMANTIC_FREEZE.json"))
    args = ap.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("schema") != SELECTION_SCHEMA:
        raise SystemExit("wrong winner-selection schema")

    kind = str(selection.get("behavior_kind", ""))
    if kind not in KIND_CONTRACT:
        raise SystemExit(f"unsupported behavior_kind: {kind!r}")
    contract = KIND_CONTRACT[kind]

    evidence_rel = str(selection.get("evidence_path", ""))
    if not evidence_rel.startswith("validation/") or not evidence_rel.endswith(".json"):
        raise SystemExit("evidence_path must be a validation/*.json repository path")
    evidence_path = Path(evidence_rel)
    if not evidence_path.is_file():
        raise SystemExit(f"evidence file not found: {evidence_rel}")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("runner_failed_before_report"):
        raise SystemExit("candidate evidence contains runner failure marker")
    if evidence.get("schema") != contract["schema"]:
        raise SystemExit("candidate evidence schema does not match selected behavior kind")

    if evidence.get("deck_formula") != DECK_FORMULA:
        raise SystemExit("candidate does not use frozen generation-2 deck formula")
    if evidence.get("deck_semantics") != "GENERATION2_AUTHORITATIVE_GLOBAL_ROOT_FORMULA_EXACT":
        raise SystemExit("candidate deck semantic is not authoritative")
    if evidence.get("extra_members_perturb_primary_rng") is not False:
        raise SystemExit("candidate side members perturb the authoritative primary RNG")
    if int(evidence.get("exact_opponent_levels", -1)) != 2:
        raise SystemExit("candidate exact-opponent level is not frozen level 2")
    if int(evidence.get("iterations", -1)) != 5 or int(evidence.get("roots_per_iteration", -1)) != 64:
        raise SystemExit("winner must be selected from the mandatory 5x64 durability horizon")
    if int(evidence.get("roots_per_seed", -1)) != 320:
        raise SystemExit("winner evidence roots_per_seed must equal 320")
    if evidence.get("acceptance_gate_changed") is not False:
        raise SystemExit("acceptance gate changed in evidence")
    if evidence.get("production_estimator_changed") not in (False, None):
        raise SystemExit("production estimator changed in evidence")
    if evidence.get("per_seed_fit_pass") is not True:
        raise SystemExit("per-seed fit gates did not all pass")
    if evidence.get("cross_seed_pass") is not True or evidence.get("r7_3_pass") is not True:
        raise SystemExit("candidate did not clear frozen R7.3 cross-seed gates")

    cross = evidence.get("cross_seed") or {}
    mean_tv = float(cross.get("mean_tv", float("inf")))
    p95_tv = float(cross.get("p95_tv", float("inf")))
    if mean_tv > FROZEN_GATES["cross_seed_mean_tv_max"] or p95_tv > FROZEN_GATES["cross_seed_p95_tv_max"]:
        raise SystemExit("candidate cross-seed numbers exceed frozen gates despite pass flag")

    ensemble_size = int(selection.get("ensemble_size", -1))
    if ensemble_size != int(evidence.get("ensemble_size", -2)):
        raise SystemExit("selection ensemble_size does not match evidence")
    if ensemble_size not in (1, 2, 4, 8):
        raise SystemExit("unsupported ensemble size")

    selected_params = dict(selection.get("params") or {})
    block_name = contract["parameter_block"]
    block = dict(evidence.get(block_name) or {}) if block_name else {}
    for key in contract["required_params"]:
        if key not in selected_params:
            raise SystemExit(f"selection missing required parameter {key}")
        if key not in block or not _float_equal(selected_params[key], block[key]):
            raise SystemExit(f"selection parameter {key} does not match evidence")

    source_head = str(selection.get("source_head_sha", ""))
    if len(source_head) < 12:
        raise SystemExit("source_head_sha is required")
    try:
        resolved_head = _git("rev-parse", f"{source_head}^{{commit}}")
    except subprocess.CalledProcessError as exc:
        raise SystemExit("source_head_sha is not present in repository history") from exc
    source_workflow_run = int(selection.get("source_workflow_run", 0))
    if source_workflow_run <= 0:
        raise SystemExit("source_workflow_run is required")
    source_workflow_path = str(selection.get("source_workflow_path", ""))
    if not source_workflow_path.startswith(".github/workflows/"):
        raise SystemExit("source_workflow_path is required")

    workflow_text = _show(resolved_head, source_workflow_path)
    _require_workflow_contract(workflow_text, ensemble_size, kind, selected_params)

    tracked_objects = {
        "python_tree": _object_sha(resolved_head, "python"),
        "src_tree": _object_sha(resolved_head, "src"),
        "include_tree": _object_sha(resolved_head, "include"),
        "tools_tree": _object_sha(resolved_head, "tools"),
        "cmake": _object_sha(resolved_head, "CMakeLists.txt"),
        "source_workflow": _object_sha(resolved_head, source_workflow_path),
        "selected_runner": _object_sha(resolved_head, contract["runner"]),
        "base_ensemble_runner": _object_sha(resolved_head, "tools/run_r7_3_partial_exact_ensemble_paired.py"),
        "partial_exact_collector": _object_sha(resolved_head, "tools/run_r7_3_partial_exact_advantage_screen.py"),
    }

    freeze = {
        "schema": FREEZE_SCHEMA,
        "selection_schema": SELECTION_SCHEMA,
        "label": str(selection.get("label", "")),
        "behavior_kind": kind,
        "behavior_semantic_id": {
            "uncertainty_damping": "SPINCORE_R7_3_UNCERTAINTY_POLICY_MIXTURE_V1",
            "temporal_blend": "SPINCORE_R7_3_TEMPORAL_POLICY_MIXTURE_V1",
            "policy_mixture": "SPINCORE_R7_3_POLICY_MIXTURE_V1",
        }[kind],
        "ensemble_size": ensemble_size,
        "params": selected_params,
        "execution_contract": dict(EXECUTION_CONTRACT),
        "roots_per_seed": 320,
        "deck_formula": DECK_FORMULA,
        "primary_rng_contract": "ONE_PERSISTENT_LIVE_BUNDLE_BATCH_RNG_IN_EXECUTION_ORDER",
        "frozen_gates": dict(FROZEN_GATES),
        "evidence_path": evidence_rel,
        "evidence_sha256": _sha256(evidence_path),
        "evidence_cross_seed": {
            "mean_tv": mean_tv,
            "p50_tv": float(cross.get("p50_tv", float("nan"))),
            "p95_tv": p95_tv,
            "max_tv": float(cross.get("max_tv", float("nan"))),
        },
        "evidence_per_seed_fit_pass": True,
        "evidence_r7_3_pass": True,
        "source_workflow_run": source_workflow_run,
        "source_workflow_path": source_workflow_path,
        "source_head_sha": resolved_head,
        "source_git_objects": tracked_objects,
        "candidate_checkpoint_extra_schema": "SPINCORE_R7_CANDIDATE_BEHAVIOR_V1",
        "acceptance_gate_changed": False,
        "ready_for_640": False,
        "ready_for_tables": False,
        "next_required_gates": [
            "FRESH_PROCESS_REPRODUCIBILITY",
            "CONTINUOUS_VS_STOP_RESTORE_CONTINUE_RECERTIFICATION",
        ],
    }
    if not freeze["label"]:
        raise SystemExit("selection label is required")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(freeze, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
