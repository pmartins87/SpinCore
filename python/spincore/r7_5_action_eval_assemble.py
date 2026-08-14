from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from spincore.r7_5_action_aggregate import (
    CandidateSelectionEvidence,
    DomainSelectionEvidence,
    prune_action_level,
)
from spincore.r7_5_action_evidence import conservative_domain_cost, learning_eligibility
from spincore.r7_5_action_stage_contract import PAIRED_EVALUATION_SEEDS, POSTFLOP_TRAINING_SEEDS
from spincore.r7_5_eval_artifacts import (
    EXPECTED_EXECUTION_SHA,
    OMISSION_COUNT_PER_CELL,
    CandidateCellEvidence,
    expected_candidate_crossplay_samples,
    load_candidate_cell_evidence,
)

ROOT_LEVEL = 160
DOMAINS = ("TRUE_HEADS_UP", "THREE_HANDED")
ELIGIBLE_CANDIDATES = (
    "PF0_CONTROL_33_75_AI",
    "PF1_33_50_75_AI",
    "PF2_33_50_75_100_AI",
    "PF3_COMPACT_33_66_100_AI",
    "PF4_CRUSHER_COMPACT_40_66_100_AI",
)
REFEREE = "PF_DENSE_REFERENCE"
ALL_CANDIDATES = (*ELIGIBLE_CANDIDATES, REFEREE)
CROSS_SEED_SCHEMA = "SPINCORE_R7_5_4A_CROSS_SEED_POLICY_STABILITY_V1"
RESULT_SCHEMA = "SPINCORE_R7_5_4A_160_RESULT_V1"


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires samples")
    return float(sum(float(value) for value in values) / len(values))


def load_cross_seed_report(path: str | Path) -> dict:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("schema") != CROSS_SEED_SCHEMA:
        raise ValueError("wrong R7.5.4A cross-seed report schema")
    if report.get("execution_sha") != EXPECTED_EXECUTION_SHA:
        raise ValueError("cross-seed report execution SHA mismatch")
    candidate = str(report.get("candidate_id"))
    domain = str(report.get("domain"))
    if candidate not in ALL_CANDIDATES or domain not in DOMAINS:
        raise ValueError("cross-seed report candidate/domain mismatch")
    if bool(report.get("production_training_authorized")) or bool(report.get("ready_for_tables")):
        raise ValueError("cross-seed report illegally authorizes production/table use")
    seed_reports = tuple(report.get("seed_reports") or ())
    if len(seed_reports) != len(POSTFLOP_TRAINING_SEEDS):
        raise ValueError("cross-seed report must embed exactly three final seed reports")
    return report


def _cell_key(value: CandidateCellEvidence) -> tuple[str, str, int, int]:
    return (
        str(value.candidate_id),
        str(value.domain),
        int(value.training_seed),
        int(value.evaluation_seed),
    )


def index_candidate_cells(
    values: Sequence[CandidateCellEvidence],
    *,
    exact_counts: bool = True,
) -> dict[tuple[str, str, int, int], CandidateCellEvidence]:
    from spincore.r7_5_eval_artifacts import validate_candidate_cell_evidence

    indexed: dict[tuple[str, str, int, int], CandidateCellEvidence] = {}
    for value in values:
        validate_candidate_cell_evidence(value, exact_counts=exact_counts)
        key = _cell_key(value)
        if key in indexed:
            raise ValueError(f"duplicate candidate cell evidence: {key}")
        indexed[key] = value
    expected = {
        (candidate, domain, int(training_seed), int(evaluation_seed))
        for candidate in ELIGIBLE_CANDIDATES
        for domain in DOMAINS
        for training_seed in POSTFLOP_TRAINING_SEEDS
        for evaluation_seed in PAIRED_EVALUATION_SEEDS
    }
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise ValueError(f"candidate cell evidence matrix mismatch missing={missing} extra={extra}")
    return indexed


def index_cross_seed_reports(values: Sequence[Mapping]) -> dict[tuple[str, str], dict]:
    indexed: dict[tuple[str, str], dict] = {}
    for raw in values:
        report = dict(raw)
        if report.get("schema") != CROSS_SEED_SCHEMA:
            raise ValueError("wrong cross-seed report schema")
        if report.get("execution_sha") != EXPECTED_EXECUTION_SHA:
            raise ValueError("cross-seed report execution SHA mismatch")
        key = (str(report.get("candidate_id")), str(report.get("domain")))
        if key in indexed:
            raise ValueError(f"duplicate cross-seed report: {key}")
        indexed[key] = report
    expected = {(candidate, domain) for candidate in ALL_CANDIDATES for domain in DOMAINS}
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise ValueError(f"cross-seed report matrix mismatch missing={missing} extra={extra}")
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
    if exact_counts:
        cell_count = len(POSTFLOP_TRAINING_SEEDS) * len(PAIRED_EVALUATION_SEEDS)
        omission_count = cell_count * OMISSION_COUNT_PER_CELL
        crossplay_count = cell_count * expected_candidate_crossplay_samples(domain)
    else:
        # Synthetic/unit-test mode uses one canonical scalar per cell while
        # preserving candidate pairing dimensions and ordering.
        cell_count = len(POSTFLOP_TRAINING_SEEDS) * len(PAIRED_EVALUATION_SEEDS)
        omission_count = cell_count
        crossplay_count = cell_count
    return (0.0,) * omission_count, (0.0,) * crossplay_count


def assemble_selection_evidence(
    *,
    candidate_cells: Sequence[CandidateCellEvidence],
    cross_seed_reports: Sequence[Mapping],
    exact_counts: bool = True,
) -> tuple[dict[str, CandidateSelectionEvidence], dict]:
    cells = index_candidate_cells(candidate_cells, exact_counts=exact_counts)
    cross = index_cross_seed_reports(cross_seed_reports)
    evidence: dict[str, CandidateSelectionEvidence] = {}
    summaries: dict[str, dict] = {}

    for candidate_id in ALL_CANDIDATES:
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
                raise ValueError("assembled selection evidence contains empty samples")
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
            "learning_gate_pass_both_domains": bool(learning_pass),
            "domains": candidate_summary,
        }
    return evidence, summaries


def aggregate_r7_5_4a_160(
    *,
    candidate_cells: Sequence[CandidateCellEvidence],
    cross_seed_reports: Sequence[Mapping],
    evaluator_sha: str,
    training_run_id: int,
    exact_counts: bool = True,
) -> dict:
    evidence, summaries = assemble_selection_evidence(
        candidate_cells=candidate_cells,
        cross_seed_reports=cross_seed_reports,
        exact_counts=exact_counts,
    )
    selection = prune_action_level(evidence, root_level=ROOT_LEVEL)
    if selection.get("selected_candidate") is not None:
        raise RuntimeError("R7.5.4A-160 illegally selected a final candidate")
    if selection.get("status") == "PASS_LEVEL" and int(selection.get("next_level", -1)) != 320:
        raise RuntimeError("R7.5.4A-160 PASS_LEVEL must escalate to 320")
    return {
        "schema": RESULT_SCHEMA,
        "root_level": ROOT_LEVEL,
        "training_run_id": int(training_run_id),
        "training_execution_sha": EXPECTED_EXECUTION_SHA,
        "evaluator_sha": str(evaluator_sha),
        "candidate_summaries": summaries,
        "selection": selection,
        "r7_5_4a_postflop_selected": False,
        "r7_5_4a_postflop_selected_candidate": None,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }


def discover_candidate_cells(directory: str | Path) -> tuple[CandidateCellEvidence, ...]:
    root = Path(directory)
    paths = sorted(root.rglob("evidence.pkl.gz"))
    if not paths:
        raise ValueError("no candidate cell evidence artifacts discovered")
    return tuple(load_candidate_cell_evidence(path, exact_counts=True) for path in paths)


def discover_cross_seed_reports(directory: str | Path) -> tuple[dict, ...]:
    root = Path(directory)
    paths = sorted(root.rglob("cross_seed_report.json"))
    if not paths:
        raise ValueError("no cross-seed reports discovered")
    return tuple(load_cross_seed_report(path) for path in paths)
