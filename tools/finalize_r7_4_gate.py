from __future__ import annotations

import argparse
import json
from pathlib import Path

import summarize_r7_4_heldout_screen as screen_mod


SCHEMA = "SPINCORE_R7_4_FINAL_GATE_V1"
DOMAIN_SCHEMA = "SPINCORE_R7_4_HELDOUT_DOMAIN_STABILITY_V1"
SCREEN_SCHEMA = "SPINCORE_R7_4_HELDOUT_SCREEN_SUMMARY_V1"


def _load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description="Materialize the finite R7.4 gate after held-out HU and 3H confirmation")
    ap.add_argument("--freeze", type=Path, required=True)
    ap.add_argument("--ruleset-freeze", type=Path, required=True)
    ap.add_argument("--screen", type=Path, required=True)
    ap.add_argument("--three-handed-confirmation", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    freeze = _load(args.freeze)
    ruleset_freeze = _load(args.ruleset_freeze)
    screen = _load(args.screen)
    three = _load(args.three_handed_confirmation)
    if freeze.get("schema") != screen_mod.FREEZE_SCHEMA or freeze.get("evidence_r7_3_pass") is not True:
        raise SystemExit("invalid R7.3 semantic freeze")
    if ruleset_freeze.get("schema") != screen_mod.RULESET_FREEZE_SCHEMA or ruleset_freeze.get("ruleset_schema") != "SPINRULESET-4":
        raise SystemExit("invalid R7.4 ruleset freeze")
    if screen.get("schema") != SCREEN_SCHEMA:
        raise SystemExit("wrong R7.4 held-out screen schema")
    if screen.get("r7_3_certified_source_head_sha") != freeze.get("source_head_sha"):
        raise SystemExit("R7.4 screen R7.3 base source differs from frozen winner")
    if screen.get("r7_4_ruleset_source_head_sha") != ruleset_freeze.get("ruleset_extension_source_head_sha"):
        raise SystemExit("R7.4 screen rules source differs from frozen SPINRULESET-4 source")
    if screen.get("durability_evidence_commit_sha") != freeze.get("evidence_commit_sha"):
        raise SystemExit("R7.4 screen evidence provenance differs from frozen winner")
    if screen.get("r7_4_heldout_screen_pass") is not True:
        raise SystemExit("R7.4 held-out screen must pass before 3H confirmation can finalize the gate")
    try:
        screen_mod._validate_domain(
            three,
            domain="THREE_HANDED",
            roots_per_iteration=128,
            freeze=freeze,
            ruleset_freeze=ruleset_freeze,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if list(three.get("algorithm_seeds") or []) != list(screen.get("heldout_algorithm_seeds") or []):
        raise SystemExit("3H confirmation changed the precommitted held-out seed set")

    hu_pass = bool(screen.get("hu", {}).get("pass"))
    three_screen_pass = bool(screen.get("three_handed_screen", {}).get("pass"))
    three_confirm_pass = bool(three.get("r7_4_domain_stability_pass"))
    passed = bool(hu_pass and three_screen_pass and three_confirm_pass)
    payload = {
        "schema": SCHEMA,
        "behavior_semantic_id": freeze["behavior_semantic_id"],
        "r7_3_certified_source_head_sha": freeze["source_head_sha"],
        "r7_4_ruleset_schema": "SPINRULESET-4",
        "r7_4_ruleset_source_head_sha": ruleset_freeze["ruleset_extension_source_head_sha"],
        "durability_evidence_commit_sha": freeze["evidence_commit_sha"],
        "heldout_algorithm_seeds": list(screen["heldout_algorithm_seeds"]),
        "r7_3_selection_seeds_reused": False,
        "gates": {
            "advantage_weighted_nrmse_max": 0.75,
            "policy_weighted_mean_tv_max": 0.12,
            "cross_seed_mean_tv_max": 0.15,
            "cross_seed_p95_tv_max": 0.35,
        },
        "hu_heldout_640_pass": hu_pass,
        "three_handed_heldout_320_screen_pass": three_screen_pass,
        "three_handed_heldout_640_confirmation": {
            "roots_per_seed": int(three["roots_per_seed"]),
            "cross_seed": dict(three["cross_seed"]),
            "per_seed_fit_pass": bool(three["per_seed_fit_pass"]),
            "scenario_coverage_pass": bool(three["scenario_coverage_pass"]),
            "pass": three_confirm_pass,
        },
        "r7_4_pass": passed,
        "r7_4_ready_to_advance_to_r8": passed,
        "acceptance_gate_changed": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
