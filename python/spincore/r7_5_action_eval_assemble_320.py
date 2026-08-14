from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from spincore.r7_5_action_320_contract import (
    REFEREE,
    ROOT_LEVEL_320,
    ExecutionPlan320,
    execution_plan_from_160_result,
)
from spincore.r7_5_action_aggregate import (
    CandidateSelectionEvidence,
    DomainSelectionEvidence,
    prune_action_level,
)
from spincore.r7_5_action_evidence import conservative_domain_cost, learning_eligibility
from spincore.r7_5_action_stage_contract import PAIRED_EVALUATION_SEEDS, POSTFLOP_TRAINING_SEEDS
from spincore.r7_5_eval_artifacts import (
    OMISSION_COUNT_PER_CELL,
    CandidateCellEvidence,
    expected_candidate_crossplay_samples,
    load_candidate_cell_evidence,
    validate_candidate_cell_evidence,
)

DOMAINS = ("TRUE_HEADS_UP", "THREE_HANDED")
CROSS_SEED_SCHEMA = "SPINCORE_R7_5_4A_CROSS_SEED_POLICY_STABILITY_V1"
RESULT_SCHEMA_320 = "SPINCORE_R7_5_4A_320_RESULT_V1"


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires samples")
    return float(sum(float(value) for value in values) / len(values))


def _validate_execution_sha(value: str) -> str:
    sha = str(value)
    if len(sha) != 40 or any(ch not in "0123456789abcdef" for ch in sha):
        raise ValueError("320 evaluator execution SHA must be a lowercase 40-hex git SHA")
    return sha


def _execution_ids(plan: ExecutionPlan320) -> tuple[str, ...]:
    ids = tuple(row.candidate_id for row in plan.execution_candidates)
    if REFEREE not in ids:
        raise ValueError("320 execution plan is missing PF_DENSE_REFERENCE")
    if len(set(ids)) != len(ids):
        raise ValueError("320 execution plan contains duplicate candidate ids")
    return ids


def _candidate_ids(plan: ExecutionPlan320) -> tuple[str, ...]:
    return tuple(candidate_id for candidate_id in _execution_ids(plan) if candidate_id != REFEREE)


def validate_cross_seed_report_320(
    report: Mapping,
    *,
    plan: ExecutionPlan320,
    expected_execution_sha: str,
) -> dict:
    row = dict(report)
    required_sha = _validate_execution_sha(expected_execution_sha)
    if row.get("schema") != CROSS_SEED_SCHEMA:
        raise ValueError("wrong R7.5.4A cross-seed report schema")
    if str(row.get("execution_sha")) != required_sha:
        raise ValueError("320 cross-seed report execution SHA mismatch")
    candidate = str(row.get("candidate_id"))
    domain = str(row.get("domain"))
    if candidate not in _execution_ids(plan) or domain not in DOMAINS:
        raise ValueError("320 cross-seed report candidate/domain mismatch")
    if bool(row.get("production_training_authorized")) or bool(row.get("ready_for_tables")):
        raise ValueError("320 cross-seed report illegally authorizes production/table use")
    seed_reports = tuple(row.get("seed_reports") or ())
    if len(seed_reports) != len(POSTFLOP_TRAINING_SEEDS):
        raise ValueError("320 cross-seed report must embed exactly three final seed reports")
    for seed_report in seed_reports:
        if str(seed_report.get("candidate_id")) != candidate:
            raise ValueError("320 cross-seed embedded candidate identity mismatch")
        if str(seed_report.get("domain")) != domain:
            raise ValueError("320 cross-seed embedded domain identity mismatch")
        if int(seed_report.get("roots", -1)) != ROOT_LEVEL_320:
            raise ValueError("320 cross-seed embedded root-level mismatch")
    return row


def index_candidate_cells_320(
    values: Sequence[CandidateCellEvidence],
    *,
    plan: ExecutionPlan320,
    expected_execution_sha: str,
    exact_counts: bool = True,
) -> dict[tuple[str, str, int, int], CandidateCellEvidence]:
    required_sha = _validate_execution_sha(expected_execution_sha)
    indexed: dict[tuple[str, str, int, int], CandidateCellEvidence] = {}
    for value in values:
        validate_candidate_cell_evidence(
            value,
            exact_counts=exact_counts,
            expected_execution_sha=required_sha,
        )
        key = (
            str(value.candidate_id),
            str(value.domain),
            int(value.training_seed),
            int(value.evaluation_seed),
        )
        if key in indexed:
            raise ValueError(f"duplicate 320 candidate cell evidence: {key}")
        indexed[key] = value
    expected = {
        (candidate, domain, int(training_seed), int(evaluation_seed))
        for candidate in _candidate_ids(plan)
        for domain in DOMAINS
        for training_seed in POSTFLOP_TRAINING_SEEDS
        for evaluation_seed in PAIRED_EVALUATION_SEEDS
    }
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise ValueError(f"320 candidate cell evidence matrix mismatch missing={missing} extra={extra}")
    return indexed


def index_cross_seed_reports_320(
    values: Sequence[Mapping],
    *,
    plan: ExecutionPlan320,
    expected_execution_sha: str,
) -> dict[tuple[str, str], dict]:
    indexed: dict[tuple[str, str], dict] = {}
    for raw in values:
        report = validate_cross_seed_report_320(
            raw,
            plan=plan,
            expected_execution_sha=expected_execution_sha,
        )
        key = (str(report["candidate_id"]), str(report["domain"]))
        if key in indexed:
            raise ValueError(f"duplicate 320 cross-seed report: {key}")
        indexed[key] = report
    expected = {(candidate, domain) for candidate in _execution_ids(plan) for domain in DOMAINS}
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise ValueError(f"320 cross-seed report matrix mismatch missing={missing} extra={extra}")
    return indexed


def _pooled_candidate_samples(
    indexed: Mapping[tuple[str, str, int, int], CandidateCellEvidence],
    *,
    candidate_id: str,
    domain: str,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    omission: list[float] = []
    crossplay: list[float] = []
    for training_seed in POSTFLOP_TRAINING_SEEDS:
        for evaluation_seed in PAIRED_EVALUATION_SEEDS:
            cell = indexed[(candidate_id, domain, int(training_seed), int(evaluation_seed))]
            omission.extend(float(value) for value in cell.omission_samples)
            crossplay.extend(float(value) for value in cell.crossplay_samples)
    return tuple(omission), tuple(crossplay)


def _dense_zero_samples(domain: str, *, exact_counts: bool) -> tuple[tuple[float, ...], tuple[float, ...]]:
    cell_count = len(POSTFLOP_TRAINING_SEEDS) * len(PAIRED_EVALUATION_SEEDS)
    if exact_counts:
        omission_count = cell_count * OMISSION_COUNT_PER_CELL
        crossplay_count = cell_count * expected_candidate_crossplay_samples(domain)
    else:
        omission_count = cell_count
        crossplay_count = cell_count
    return (0.0,) * omission_count, (0.0,) * crossplay_count


def assemble_selection_evidence_320(
    *,
    plan: ExecutionPlan320,
    candidate_cells: Sequence[CandidateCellEvidence],
    cross_seed_reports: Sequence[Mapping],
    expected_execution_sha: str,
    exact_counts: bool = True,
) -> tuple[dict[str, CandidateSelectionEvidence], dict]:
    cells = index_candidate_cells_320(
        candidate_cells,
        plan=plan,
        expected_execution_sha=expected_execution_sha,
        exact_counts=exact_counts,
    )
    cross = index_cross_seed_reports_320(
        cross_seed_reports,
        plan=plan,
        expected_execution_sha=expected_execution_sha,
    )
    evidence: dict[str, CandidateSelectionEvidence] = {}
    summaries: dict[str, dict] = {}

    for candidate_id in _execution_ids(plan):
        domains: dict[str, DomainSelectionEvidence] = {}
        domain_learning: dict[str, bool] = {}
        candidate_summary: dict[str, dict] = {}
        for domain in DOMAINS:
            report = cross[(candidate_id, domain)]
            seed_reports = tuple(report.get("seed_reports") or ())
            cost = conservative_domain_cost(
                seed_reports,
                candidate_id=candidate_id,
                domain=domain,
            )
            domain_learning[domain] = learning_eligibility(
                seed_reports,
                candidate_id=candidate_id,
                domain=domain,
                cross_seed_report=report,
            )
            if candidate_id == REFEREE:
                omission, crossplay = _dense_zero_samples(domain, exact_counts=exact_counts)
            else:
                omission, crossplay = _pooled_candidate_samples(
                    cells,
                    candidate_id=candidate_id,
                    domain=domain,
                )
            if not omission or not crossplay:
                raise ValueError("assembled 320 selection evidence contains empty samples")
            domains[domain] = DomainSelectionEvidence(
                omission_samples=omission,
                crossplay_samples=crossplay,
                nodes_per_root=float(cost.nodes_per_root),
                tree_seconds_per_root=float(cost.tree_seconds_per_root),
                effective_branches_per_decision=float(cost.effective_branches_per_decision),
            )
            candidate_summary[domain] = {
                "learning_gate_pass": bool(domain_learning[domain]),
                "omission_sample_count": len(omission),
                "omission_mean": _mean(omission),
                "crossplay_sample_count": len(crossplay),
                "crossplay_mean": _mean(crossplay),
                "nodes_per_root_worst_seed": float(cost.nodes_per_root),
                "tree_seconds_per_root_worst_seed": float(cost.tree_seconds_per_root),
                "effective_branches_per_decision_worst_seed": float(cost.effective_branches_per_decision),
                "peak_rss_bytes_worst_seed": int(cost.peak_rss_bytes),
                "full_training_seconds_per_root_worst_seed": float(cost.full_training_seconds_per_root),
                "cross_seed_mean_tv": float(report["mean_tv"]),
                "cross_seed_p95_tv": float(report["p95_tv"]),
                "cross_seed_gate_pass": bool(report["gate_pass"]),
            }
        learning_pass = all(domain_learning.values())
        evidence[candidate_id] = CandidateSelectionEvidence(
            candidate_id=candidate_id,
            learning_gate_pass=bool(learning_pass),
            domains=domains,
        )
        summaries[candidate_id] = {
            "strategically_eligible_at_320": bool(candidate_id in plan.survivors),
            "learning_gate_pass_both_domains": bool(learning_pass),
            "domains": candidate_summary,
        }
    return evidence, summaries


def aggregate_r7_5_4a_320(
    *,
    parent_160_result: Mapping,
    candidate_cells: Sequence[CandidateCellEvidence],
    cross_seed_reports: Sequence[Mapping],
    training_execution_sha: str,
    evaluator_sha: str,
    training_run_id: int,
    exact_counts: bool = True,
) -> dict:
    required_sha = _validate_execution_sha(training_execution_sha)
    plan = execution_plan_from_160_result(parent_160_result)
    evidence, summaries = assemble_selection_evidence_320(
        plan=plan,
        candidate_cells=candidate_cells,
        cross_seed_reports=cross_seed_reports,
        expected_execution_sha=required_sha,
        exact_counts=exact_counts,
    )
    selection = prune_action_level(
        evidence,
        root_level=ROOT_LEVEL_320,
        prior_eligible_ids=plan.survivors,
    )
    selected = selection.get("selected_candidate")
    next_level = selection.get("next_level")
    if selection.get("status") == "PASS_LEVEL":
        if selected is not None:
            if str(selected) not in set(plan.survivors):
                raise RuntimeError("320 selected a candidate that did not survive 160")
            if next_level is not None:
                raise RuntimeError("320 cannot both select and escalate")
        else:
            if int(next_level or -1) != 640:
                raise RuntimeError("ambiguous 320 result must escalate fresh to 640")
    return {
        "schema": RESULT_SCHEMA_320,
        "root_level": ROOT_LEVEL_320,
        "parent_160_training_execution_sha": str(parent_160_result.get("training_execution_sha")),
        "parent_160_evaluator_sha": str(parent_160_result.get("evaluator_sha")),
        "prior_eligible_survivors": list(plan.survivors),
        "executed_candidates": list(_execution_ids(plan)),
        "training_run_id": int(training_run_id),
        "training_execution_sha": required_sha,
        "evaluator_sha": str(evaluator_sha),
        "candidate_summaries": summaries,
        "selection": selection,
        "r7_5_4a_postflop_selected": bool(selected is not None),
        "r7_5_4a_postflop_selected_candidate": str(selected) if selected is not None else None,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }


def discover_candidate_cells_320(
    directory: str | Path,
    *,
    expected_execution_sha: str,
) -> tuple[CandidateCellEvidence, ...]:
    root = Path(directory)
    paths = sorted(root.rglob("evidence.pkl.gz"))
    if not paths:
        raise ValueError("no 320 candidate cell evidence artifacts discovered")
    return tuple(
        load_candidate_cell_evidence(
            path,
            exact_counts=True,
            expected_execution_sha=expected_execution_sha,
        )
        for path in paths
    )


def discover_cross_seed_reports_320(
    directory: str | Path,
    *,
    plan: ExecutionPlan320,
    expected_execution_sha: str,
) -> tuple[dict, ...]:
    root = Path(directory)
    paths = sorted(root.rglob("cross_seed_report.json"))
    if not paths:
        raise ValueError("no 320 cross-seed reports discovered")
    return tuple(
        validate_cross_seed_report_320(
            json.loads(path.read_text(encoding="utf-8")),
            plan=plan,
            expected_execution_sha=expected_execution_sha,
        )
        for path in paths
    )
