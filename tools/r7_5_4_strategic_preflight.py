from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA = "SPINCORE_R7_5_4_STRATEGIC_PREFLIGHT_V5"
REP_SCHEMA = "SPINCORE_R7_5_3_REPRESENTATION_ABLATION_RESULT_V1"
STRUCT_SCHEMA = "SPINCORE_R7_5_4_ACTION_STRUCTURAL_AUDIT_V3"
UNCERTAINTY_SCHEMA = "SPINCORE_R7_5_4_UNCERTAINTY_EQUIVALENCE_AUDIT_V1"
TRAINING_FREEZE_SCHEMA = "SPINCORE_R7_5_4_TRAINING_IMPLEMENTATION_FREEZE_V1"
PRECOMMIT_V1_SCHEMA = "SPINCORE_R7_5_4_ACTION_ABSTRACTION_ABLATION_PRECOMMIT_V1"
PRECOMMIT_V2_SCHEMA = "SPINCORE_R7_5_4_ACTION_ABSTRACTION_ABLATION_PRECOMMIT_V2"
PRECOMMIT_V3_SCHEMA = "SPINCORE_R7_5_4_ACTION_ABSTRACTION_ABLATION_PRECOMMIT_V3"
HERITAGE_SCHEMA = "SPINCORE_LEGACY_HERITAGE_SOURCE_MANIFEST_V1"
HERITAGE_LEDGER_SCHEMA = "SPINCORE_R7_5_LEGACY_HERITAGE_INTEGRATION_LEDGER_V1"
REACHABILITY_SCHEMA = "SPINCORE_R7_5_REAL_GAME_REACHABILITY_CONTRACT_V1"
REACHABILITY_AUDIT_SCHEMA = "SPINCORE_R7_5_REAL_GAME_REACHABILITY_AUDIT_V1"
TRAINABILITY_SCHEMA = "SPINCORE_R8_TRAINABILITY_TIME_BUDGET_CONTRACT_V2"
TRAINABILITY_PROJECTION_SCHEMA = "SPINCORE_R8_PRODUCTION_TRAINABILITY_V2"

HERITAGE_AUDIT = "R7_5_FULL_PRIOR_ATTEMPT_HERITAGE_AUDIT_20260814.md"
HERITAGE_MANIFEST = "R7_5_LEGACY_HERITAGE_SOURCE_MANIFEST_20260814.json"
HERITAGE_LEDGER = "R7_5_LEGACY_HERITAGE_INTEGRATION_LEDGER_20260814.json"
REACHABILITY_CONTRACT = "R7_5_REAL_GAME_REACHABILITY_CONTRACT.json"
REACHABILITY_AUDIT = "R7_5_REAL_GAME_REACHABILITY_AUDIT.json"
TRAINABILITY_CONTRACT = "R8_TRAINABILITY_TIME_BUDGET_CONTRACT_20260814.json"

REPRESENTATIONS = {
    "C0_V1_FROZEN_CONTROL",
    "C1_V2_NO_FLOP_TOKEN",
    "C2_V2_H1_CANONICAL_184",
    "C3_V2_H2_MIN_CHANGE_181",
    "C4_V2_H3_RECLUSTERED_184",
    "C5_V2_H4_EXACT_1755",
}

REQUIRED_REACHABILITY_INVARIANTS = {
    "legal_hand_start",
    "legal_seat_dealer_blind_assignment",
    "unique_cards_compatible_with_street",
    "folded_player_never_acts_again",
    "allin_player_never_takes_later_voluntary_action",
    "current_actor_is_exact_next_legal_actor",
    "ordered_public_history_reaches_exact_current_contributions",
    "pot_reconciles_with_history_and_contributions",
    "amount_to_call_from_exact_state",
    "min_and_max_raise_from_exact_betting_engine",
    "legal_action_mask_from_exact_engine",
    "live_folded_allin_counts_agree_with_player_status",
    "position_and_ip_oop_agree_with_dealer_and_live_lineup",
    "preflop_lineage_derived_from_ordered_public_events",
    "initiative_and_postflop_line_derived_from_ordered_public_events",
    "hand_draw_board_semantics_deterministic_from_actual_cards",
    "derived_features_cannot_contradict_authoritative_state",
    "snapshot_only_reconstruction_for_path_dependent_semantics_forbidden",
}

REQUIRED_REACHABILITY_CLAIMS = {
    "observations_only_from_engine_reached_states",
    "spnniv1_legal_mask_matches_engine_on_audited_states",
    "spnniv2_legal_mask_matches_engine_on_audited_states",
    "illegal_actions_fail_closed_on_audited_states",
    "terminal_chip_and_icm_conservation_on_audited_states",
    "universal_action_path_traversed",
    "legacy_action_path_traversed",
}


def _read(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(repo_root: str | Path, *, phase: str, root_level: int) -> dict:
    root = Path(repo_root)
    validation = root / "validation"
    checks: dict[str, dict] = {}

    def check(name: str, condition: bool, detail: str) -> None:
        checks[name] = {"pass": bool(condition), "detail": str(detail)}

    readable_source_names: set[str] = set()

    try:
        heritage = _read(validation / HERITAGE_MANIFEST)
        archive = heritage.get("prior_attempt_archive") or {}
        comparison = archive.get("archive_comparison") or {}
        readable = archive.get("readable_sources") or []
        readable_source_names = {str(row.get("source")) for row in readable}
        check("heritage_schema", heritage.get("schema") == HERITAGE_SCHEMA, str(heritage.get("schema")))
        check(
            "heritage_full_read_inventory",
            len(readable) == 17
            and len(readable_source_names) == 17
            and all(
                str(row.get("full_read_status", "")).endswith(("full_read", "full_parse", "ast_parse"))
                or "full_" in str(row.get("full_read_status", ""))
                for row in readable
            ),
            f"readable_sources={len(readable)} unique={len(readable_source_names)}",
        )
        check(
            "heritage_archive_equivalence",
            int(comparison.get("shared_entries", -1)) == 27
            and int(comparison.get("shared_byte_identical", -1)) == 27
            and set(comparison.get("only_in_superset") or [])
            == {"hardcoded/Crusher Framework 5.txt", "solver v2/184Flops.json"},
            json.dumps(comparison, sort_keys=True),
        )
        check(
            "heritage_not_table_authority",
            not bool(heritage.get("ready_for_tables")),
            "heritage audit cannot authorize table use",
        )
    except Exception as exc:
        check("heritage_manifest_available", False, repr(exc))

    heritage_audit_path = validation / HERITAGE_AUDIT
    check(
        "heritage_audit_available",
        heritage_audit_path.exists() and heritage_audit_path.stat().st_size > 0,
        str(heritage_audit_path),
    )

    try:
        ledger = _read(validation / HERITAGE_LEDGER)
        entries = ledger.get("entries") or []
        ledger_sources = {str(row.get("source")) for row in entries}
        complete_rows = all(
            bool(row.get("reviewed_full"))
            and bool(row.get("best_of"))
            and bool(row.get("preserve"))
            and bool(str(row.get("destination", "")).strip())
            and bool(str(row.get("status", "")).strip())
            and bool(row.get("current_evidence"))
            and bool(row.get("never_inherit_as_truth"))
            for row in entries
        )
        check(
            "heritage_ledger_schema",
            ledger.get("schema") == HERITAGE_LEDGER_SCHEMA,
            str(ledger.get("schema")),
        )
        check(
            "heritage_ledger_complete",
            len(entries) == 17
            and len(ledger_sources) == 17
            and ledger_sources == readable_source_names
            and complete_rows,
            f"entries={len(entries)} ledger_sources={len(ledger_sources)} manifest_sources={len(readable_source_names)}",
        )
        check(
            "heritage_ledger_not_table_authority",
            not bool(ledger.get("strategic_output"))
            and not bool(ledger.get("production_training_authorized"))
            and not bool(ledger.get("ready_for_tables")),
            "per-file heritage disposition cannot authorize strategy/table use",
        )
    except Exception as exc:
        check("heritage_ledger_available", False, repr(exc))

    try:
        reach = _read(validation / REACHABILITY_CONTRACT)
        invariants = reach.get("required_invariants") or {}
        check("reachability_schema", reach.get("schema") == REACHABILITY_SCHEMA, str(reach.get("schema")))
        missing = sorted(k for k in REQUIRED_REACHABILITY_INVARIANTS if not bool(invariants.get(k)))
        check("reachable_state_contract", not missing, f"missing_or_false={missing}")
        check(
            "reachability_not_table_authority",
            not bool(reach.get("ready_for_tables")) and not bool(reach.get("production_training_authorized")),
            "semantic contract is pre-output only",
        )
    except Exception as exc:
        check("reachability_contract_available", False, repr(exc))

    try:
        reach_audit = _read(validation / REACHABILITY_AUDIT)
        totals = reach_audit.get("totals") or {}
        claims = reach_audit.get("claims") or {}
        check(
            "reachability_audit_schema",
            reach_audit.get("schema") == REACHABILITY_AUDIT_SCHEMA,
            str(reach_audit.get("schema")),
        )
        missing_claims = sorted(k for k in REQUIRED_REACHABILITY_CLAIMS if not bool(claims.get(k)))
        check(
            "reachability_audit_gate",
            bool(reach_audit.get("reachability_gate_pass"))
            and int(totals.get("trajectories", 0)) == 60
            and int(totals.get("v1_legal_mask_checks", 0)) > 0
            and int(totals.get("v2_legal_mask_checks", 0)) > 0
            and int(totals.get("legacy_illegal_rejections", 0)) > 0
            and int(totals.get("universal_illegal_rejections", 0)) > 0
            and int(totals.get("terminal_chip_conservation_checks", 0)) == 60
            and int(totals.get("terminal_icm_conservation_checks", 0)) == 60
            and not missing_claims,
            f"totals={json.dumps(totals,sort_keys=True)} missing_claims={missing_claims}",
        )
        check(
            "reachability_audit_not_table_authority",
            not bool(reach_audit.get("ready_for_tables"))
            and not bool(reach_audit.get("production_training_authorized")),
            "reachability audit is mechanism evidence only",
        )
    except Exception as exc:
        check("reachability_audit_available", False, repr(exc))

    try:
        rep = _read(validation / "R7_5_3_REPRESENTATION_ABLATION_RESULT.json")
        check("representation_schema", rep.get("schema") == REP_SCHEMA, str(rep.get("schema")))
        check(
            "representation_gate",
            bool(rep.get("r7_5_3_representation_ablation_pass")),
            f"selected={rep.get('selected_candidate')}",
        )
        selected_representation = rep.get("selected_candidate")
        check(
            "representation_selected",
            selected_representation in REPRESENTATIONS,
            str(selected_representation),
        )
        check(
            "representation_not_table_authority",
            not bool(rep.get("ready_for_tables")) and not bool(rep.get("production_training_authorized")),
            "R7.5.3 must authorize representation only",
        )
    except Exception as exc:
        selected_representation = None
        check("representation_result_available", False, repr(exc))

    # Trainability is precommitted before action-sizing outputs.  R7.5.4 does not
    # require the future physical R8 PASS yet, because the exact R8.0 profile and
    # Ryzen R8.2 calibration are still prerequisites.  It *does* require the hard
    # 90-day contract to exist unchanged and to remain explicitly NOT MEASURED / NOT PASS.
    try:
        trainability = _read(validation / TRAINABILITY_CONTRACT)
        hard_cap = trainability.get("hard_cap") or {}
        reserve = trainability.get("planning_reserve") or {}
        projection = trainability.get("projection_contract") or {}
        rep_state = trainability.get("representation_state_at_freeze") or {}
        physical = trainability.get("physical_measurement_contract") or {}
        check(
            "trainability_contract_schema",
            trainability.get("schema") == TRAINABILITY_SCHEMA,
            str(trainability.get("schema")),
        )
        check(
            "trainability_budget_frozen",
            float(hard_cap.get("wall_clock_days", -1.0)) == 90.0
            and float(reserve.get("multiplier", -1.0)) == 1.20
            and float(reserve.get("implied_nominal_budget_days", -1.0)) == 75.0
            and projection.get("schema") == TRAINABILITY_PROJECTION_SCHEMA
            and projection.get("required_domains") == ["TRUE_HEADS_UP", "THREE_HANDED"]
            and bool(projection.get("all_selected_profiles_required"))
            and bool(projection.get("all_required_algorithm_seed_streams_required"))
            and bool(projection.get("all_frozen_iterations_required"))
            and bool(projection.get("all_non_iteration_training_and_freeze_work_required"))
            and physical.get("timing_cost_basis") == "MATURE_OR_WORST_CASE_CERTIFIED_V1"
            and physical.get("non_iteration_scope") == "ALL_FROZEN_NON_ITERATION_TRAINING_AND_FINAL_FREEZE_WORK_V1",
            f"cap={hard_cap.get('wall_clock_days')} reserve={reserve.get('multiplier')} projection={projection.get('schema')}",
        )
        check(
            "trainability_bound_to_representation",
            selected_representation is not None
            and rep_state.get("selected_candidate") == selected_representation
            and int(rep_state.get("serialized_observation_bytes", -1)) == 126
            and int(rep_state.get("model_parameter_count", -1)) == 152438,
            f"preflight_selected={selected_representation} contract_selected={rep_state.get('selected_candidate')}",
        )
        check(
            "trainability_pre_physical_status",
            trainability.get("current_trainability_status") == "NOT_MEASURED_PHYSICALLY / NOT_PASS"
            and not bool(trainability.get("production_training_authorized"))
            and not bool(trainability.get("ready_for_tables")),
            str(trainability.get("current_trainability_status")),
        )
    except Exception as exc:
        check("trainability_contract_available", False, repr(exc))

    try:
        structural = _read(validation / "R7_5_4_ACTION_STRUCTURAL_AUDIT.json")
        check("structural_schema", structural.get("schema") == STRUCT_SCHEMA, str(structural.get("schema")))
        check("structural_gate", bool(structural.get("structural_gate_pass")), "durable structural evidence")
        check(
            "structural_not_table_authority",
            not bool(structural.get("ready_for_tables"))
            and not bool(structural.get("production_training_authorized")),
            "structural audit is mechanism-only",
        )
    except Exception as exc:
        check("structural_evidence_available", False, repr(exc))

    try:
        uncertainty = _read(validation / "R7_5_4_UNCERTAINTY_EQUIVALENCE.json")
        check("uncertainty_schema", uncertainty.get("schema") == UNCERTAINTY_SCHEMA, str(uncertainty.get("schema")))
        check(
            "uncertainty_equivalence",
            bool(uncertainty.get("uncertainty_equivalence_pass"))
            and float(uncertainty.get("maximum_abs_difference", 1.0)) <= 1e-12,
            f"max_diff={uncertainty.get('maximum_abs_difference')}",
        )
        check(
            "uncertainty_not_table_authority",
            not bool(uncertainty.get("ready_for_tables"))
            and not bool(uncertainty.get("production_training_authorized")),
            "uncertainty audit is mechanism-only",
        )
    except Exception as exc:
        check("uncertainty_evidence_available", False, repr(exc))

    try:
        v1 = _read(validation / "R7_5_4_ACTION_ABSTRACTION_ABLATION_PRECOMMIT.json")
        v2 = _read(validation / "R7_5_4_ACTION_ABSTRACTION_ABLATION_PRECOMMIT_V2.json")
        v3 = _read(validation / "R7_5_4_ACTION_ABSTRACTION_ABLATION_PRECOMMIT_V3.json")
        freeze = _read(validation / "R7_5_4_TRAINING_IMPLEMENTATION_FREEZE.json")
        check("precommit_v1", v1.get("schema") == PRECOMMIT_V1_SCHEMA, str(v1.get("schema")))
        check("precommit_v2", v2.get("schema") == PRECOMMIT_V2_SCHEMA, str(v2.get("schema")))
        check("precommit_v3", v3.get("schema") == PRECOMMIT_V3_SCHEMA, str(v3.get("schema")))
        check("training_freeze", freeze.get("schema") == TRAINING_FREEZE_SCHEMA, str(freeze.get("schema")))
        check(
            "no_precommit_table_authority",
            all(not bool(document.get("ready_for_tables")) for document in (v1, v2, v3, freeze)),
            "all pre-output contracts preserve READY FOR TABLES=false",
        )
    except Exception as exc:
        check("precommit_chain_available", False, repr(exc))

    check("phase", phase in {"R7_5_4A_POSTFLOP", "R7_5_4B_PREFLOP"}, phase)
    check("root_level", int(root_level) in {160, 320, 640}, str(root_level))

    if phase == "R7_5_4A_POSTFLOP":
        check(
            "initial_postflop_level",
            int(root_level) == 160,
            "R7.5.4A starts at the frozen 160-root pruning level",
        )
    elif phase == "R7_5_4B_PREFLOP":
        try:
            postflop = _read(validation / "R7_5_4A_POSTFLOP_SELECTION.json")
            check(
                "postflop_selection_available",
                bool(postflop.get("selection_pass")) and postflop.get("selected_candidate"),
                str(postflop.get("selected_candidate")),
            )
        except Exception as exc:
            check("postflop_selection_available", False, repr(exc))

    passed = bool(checks) and all(row["pass"] for row in checks.values())
    return {
        "schema": SCHEMA,
        "phase": phase,
        "root_level": int(root_level),
        "selected_representation": selected_representation,
        "checks": checks,
        "ready_to_start": passed,
        "physical_trainability_pass_required_now": False,
        "physical_trainability_gate_required_before_r8_official_training": True,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed preflight for strategic R7.5.4 action training")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--phase", choices=["R7_5_4A_POSTFLOP", "R7_5_4B_PREFLOP"], required=True)
    parser.add_argument("--root-level", type=int, choices=[160, 320, 640], required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    payload = evaluate(args.repo_root, phase=args.phase, root_level=args.root_level)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if payload["ready_to_start"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
