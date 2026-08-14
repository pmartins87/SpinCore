from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from spincore.r7_5_action_stage_contract import (
    ITERATIONS,
    POSTFLOP_TRAINING_SEEDS,
    ROOT_LEVEL,
    ROOTS_PER_ITERATION,
    SELECTED_REPRESENTATION,
)

DOMAINS = ("TRUE_HEADS_UP", "THREE_HANDED")


@dataclass(frozen=True)
class ConservativeDomainCost:
    candidate_id: str
    domain: str
    nodes_per_root: float
    tree_seconds_per_root: float
    effective_branches_per_decision: float
    peak_rss_bytes: int
    full_training_seconds_per_root: float
    seed_reports_valid: bool
    per_seed_learning_gates_pass: bool


def _finite_nonnegative(report: Mapping, key: str) -> float:
    value = float(report[key])
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"invalid final-report {key}: {value!r}")
    return value


def validate_final_seed_reports(
    reports: Sequence[Mapping],
    *,
    candidate_id: str,
    domain: str,
) -> tuple[Mapping, ...]:
    if str(domain) not in DOMAINS:
        raise ValueError(f"unsupported R7.5.4A domain: {domain!r}")
    rows = tuple(reports)
    if len(rows) != len(POSTFLOP_TRAINING_SEEDS):
        raise ValueError("candidate/domain evidence requires exactly three final seed reports")
    by_seed: dict[int, Mapping] = {}
    for report in rows:
        if str(report.get("candidate_id")) != str(candidate_id):
            raise ValueError("final seed report candidate mismatch")
        if str(report.get("domain")) != str(domain):
            raise ValueError("final seed report domain mismatch")
        if str(report.get("selected_representation")) != SELECTED_REPRESENTATION:
            raise ValueError("final seed report representation mismatch")
        if int(report.get("iterations", -1)) != ITERATIONS:
            raise ValueError("final seed report iteration count mismatch")
        if int(report.get("roots_per_iteration", -1)) != ROOTS_PER_ITERATION:
            raise ValueError("final seed report roots-per-iteration mismatch")
        if int(report.get("roots", -1)) != ROOT_LEVEL:
            raise ValueError("final seed report root-level mismatch")
        if bool(report.get("strategic_selection_permitted_at_160")):
            raise ValueError("160-root report illegally permits final selection")
        if bool(report.get("production_training_authorized")) or bool(report.get("ready_for_tables")):
            raise ValueError("R7.5.4A report illegally authorizes production/table use")
        seed = int(report.get("training_seed", -1))
        if seed in by_seed:
            raise ValueError("duplicate final seed report")
        by_seed[seed] = report
        for key in (
            "nodes_per_root",
            "tree_seconds_per_root",
            "effective_unique_aggressive_branches_per_decision",
            "full_training_seconds_per_root",
        ):
            _finite_nonnegative(report, key)
        peak = int(report.get("peak_rss_bytes", -1))
        if peak < 0:
            raise ValueError("invalid final-report peak_rss_bytes")
    if set(by_seed) != set(POSTFLOP_TRAINING_SEEDS):
        raise ValueError("final seed report set differs from frozen postflop seeds")
    return tuple(by_seed[seed] for seed in POSTFLOP_TRAINING_SEEDS)


def conservative_domain_cost(
    reports: Sequence[Mapping],
    *,
    candidate_id: str,
    domain: str,
) -> ConservativeDomainCost:
    rows = validate_final_seed_reports(reports, candidate_id=candidate_id, domain=domain)
    return ConservativeDomainCost(
        candidate_id=str(candidate_id),
        domain=str(domain),
        nodes_per_root=max(_finite_nonnegative(row, "nodes_per_root") for row in rows),
        tree_seconds_per_root=max(_finite_nonnegative(row, "tree_seconds_per_root") for row in rows),
        effective_branches_per_decision=max(
            _finite_nonnegative(row, "effective_unique_aggressive_branches_per_decision")
            for row in rows
        ),
        peak_rss_bytes=max(int(row["peak_rss_bytes"]) for row in rows),
        full_training_seconds_per_root=max(
            _finite_nonnegative(row, "full_training_seconds_per_root") for row in rows
        ),
        seed_reports_valid=True,
        per_seed_learning_gates_pass=all(
            bool(row.get("advantage_gate_pass")) and bool(row.get("policy_gate_pass"))
            for row in rows
        ),
    )


def learning_eligibility(
    reports: Sequence[Mapping],
    *,
    candidate_id: str,
    domain: str,
    cross_seed_report: Mapping,
) -> bool:
    cost = conservative_domain_cost(
        reports,
        candidate_id=candidate_id,
        domain=domain,
    )
    if str(cross_seed_report.get("candidate_id")) != str(candidate_id):
        raise ValueError("cross-seed report candidate mismatch")
    if str(cross_seed_report.get("domain")) != str(domain):
        raise ValueError("cross-seed report domain mismatch")
    return bool(cost.per_seed_learning_gates_pass and cross_seed_report.get("gate_pass"))
