from __future__ import annotations

import argparse
import json
from pathlib import Path


SCHEMA = "SPINCORE_R7_4_HELDOUT_SCREEN_SUMMARY_V1"
DOMAIN_SCHEMA = "SPINCORE_R7_4_HELDOUT_DOMAIN_STABILITY_V1"
FREEZE_SCHEMA = "SPINCORE_R7_3_CANDIDATE_SEMANTIC_FREEZE_V1"
ACCEPT_SCHEMA = "SPINCORE_R7_3_FROZEN_CANDIDATE_640_ACCEPTANCE_V1"
PREFLIGHT_SCHEMA = "SPINCORE_R7_4_DOMAIN_PREFLIGHT_V1"


def _load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def _validate_domain(row: dict, *, domain: str, roots_per_iteration: int, freeze: dict) -> None:
    if row.get("schema") != DOMAIN_SCHEMA:
        raise ValueError(f"{domain}: wrong domain evidence schema")
    if row.get("domain") != domain:
        raise ValueError(f"{domain}: wrong domain label")
    if row.get("accepted_r7_3_source_head_sha") != freeze.get("source_head_sha"):
        raise ValueError(f"{domain}: source head differs from frozen winner")
    if row.get("accepted_r7_3_evidence_commit_sha") != freeze.get("evidence_commit_sha"):
        raise ValueError(f"{domain}: evidence commit differs from frozen winner")
    if row.get("accepted_r7_3_behavior_semantic_id") != freeze.get("behavior_semantic_id"):
        raise ValueError(f"{domain}: behavior semantic differs from frozen winner")
    if row.get("exact_accepted_algorithm_source_used") is not True:
        raise ValueError(f"{domain}: exact accepted algorithm source not proven")
    if row.get("pilot_worker_executed_from_accepted_worktree_overlay") is not True:
        raise ValueError(f"{domain}: worker provenance incomplete")
    if row.get("thread_environment_overrides_injected_by_r7_4_orchestrator") is not False:
        raise ValueError(f"{domain}: hidden thread override detected or unproven")
    if row.get("r7_3_640_acceptance_passed_first") is not True or row.get("r7_4_structural_preflight_passed_first") is not True:
        raise ValueError(f"{domain}: prerequisite gates not proven")
    if int(row.get("iterations", -1)) != 5:
        raise ValueError(f"{domain}: expected five CFR iterations")
    if int(row.get("roots_per_iteration", -1)) != int(roots_per_iteration):
        raise ValueError(f"{domain}: wrong roots_per_iteration")
    if int(row.get("roots_per_seed", -1)) != 5 * int(roots_per_iteration):
        raise ValueError(f"{domain}: wrong roots_per_seed")
    if int(row.get("exact_opponent_levels", -1)) != 2:
        raise ValueError(f"{domain}: exact-opponent level changed")
    if row.get("deck_formula") != "seed*1000003 + global_root*97 + iteration":
        raise ValueError(f"{domain}: deck formula changed")
    if row.get("extra_members_perturb_primary_rng") is not False:
        raise ValueError(f"{domain}: side ensemble perturbs primary RNG")
    if row.get("r7_3_selection_seeds_reused") is not False:
        raise ValueError(f"{domain}: R7.3 selection seeds were reused")
    if row.get("scenario_coverage_pass") is not True:
        raise ValueError(f"{domain}: scenario coverage incomplete")
    if row.get("acceptance_gate_changed") is not False:
        raise ValueError(f"{domain}: acceptance gate changed")
    if row.get("ready_for_tables") is not False:
        raise ValueError(f"{domain}: premature table readiness")


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize the precommitted R7.4 held-out HU/3H stability screen")
    ap.add_argument("--freeze", type=Path, required=True)
    ap.add_argument("--acceptance", type=Path, required=True)
    ap.add_argument("--preflight", type=Path, required=True)
    ap.add_argument("--hu", type=Path, required=True)
    ap.add_argument("--three-handed", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    freeze = _load(args.freeze)
    acceptance = _load(args.acceptance)
    preflight = _load(args.preflight)
    hu = _load(args.hu)
    three = _load(args.three_handed)
    if freeze.get("schema") != FREEZE_SCHEMA or freeze.get("evidence_r7_3_pass") is not True:
        raise SystemExit("invalid semantic freeze")
    if acceptance.get("schema") != ACCEPT_SCHEMA or acceptance.get("r7_3_640_acceptance_pass") is not True:
        raise SystemExit("R7.3 640 acceptance missing")
    if preflight.get("schema") != PREFLIGHT_SCHEMA or preflight.get("r7_4_structural_preflight_pass") is not True:
        raise SystemExit("R7.4 structural preflight missing")
    try:
        _validate_domain(hu, domain="TRUE_HEADS_UP", roots_per_iteration=128, freeze=freeze)
        _validate_domain(three, domain="THREE_HANDED", roots_per_iteration=64, freeze=freeze)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if list(hu.get("algorithm_seeds") or []) != list(three.get("algorithm_seeds") or []):
        raise SystemExit("HU and 3H held-out seed sets differ")

    hu_pass = bool(hu.get("r7_4_domain_stability_pass"))
    three_pass = bool(three.get("r7_4_domain_stability_pass"))
    screen_pass = bool(hu_pass and three_pass)
    payload = {
        "schema": SCHEMA,
        "behavior_semantic_id": freeze["behavior_semantic_id"],
        "source_head_sha": freeze["source_head_sha"],
        "durability_evidence_commit_sha": freeze["evidence_commit_sha"],
        "heldout_algorithm_seeds": list(hu["algorithm_seeds"]),
        "r7_3_selection_seeds_reused": False,
        "hu": {
            "roots_per_seed": int(hu["roots_per_seed"]),
            "cross_seed": dict(hu["cross_seed"]),
            "per_seed_fit_pass": bool(hu["per_seed_fit_pass"]),
            "pass": hu_pass,
        },
        "three_handed_screen": {
            "roots_per_seed": int(three["roots_per_seed"]),
            "cross_seed": dict(three["cross_seed"]),
            "per_seed_fit_pass": bool(three["per_seed_fit_pass"]),
            "pass": three_pass,
        },
        "r7_4_heldout_screen_pass": screen_pass,
        "three_handed_640_confirmation_required": screen_pass,
        "r7_4_final_pass": False,
        "acceptance_gate_changed": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
