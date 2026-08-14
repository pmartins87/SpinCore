from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from spincore.r7_5_referee_rng import (
    ordered_candidate_pair,
    paired_bootstrap_mean_ci,
    paired_difference,
)

DOMAINS = ("TRUE_HEADS_UP", "THREE_HANDED")
CONTROL = "PF0_CONTROL_33_75_AI"
REFEREE = "PF_DENSE_REFERENCE"
MATERIAL_FLOOR = 0.001


@dataclass(frozen=True)
class DomainSelectionEvidence:
    omission_samples: tuple[float, ...]
    crossplay_samples: tuple[float, ...]
    nodes_per_root: float
    tree_seconds_per_root: float
    effective_branches_per_decision: float

    def __post_init__(self) -> None:
        if not self.omission_samples or not self.crossplay_samples:
            raise ValueError("selection evidence requires nonempty paired samples")
        values = (
            *self.omission_samples,
            *self.crossplay_samples,
            self.nodes_per_root,
            self.tree_seconds_per_root,
            self.effective_branches_per_decision,
        )
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("selection evidence contains non-finite values")
        if any(float(value) < 0.0 for value in self.omission_samples):
            raise ValueError("omission samples must be nonnegative")
        if min(self.nodes_per_root, self.tree_seconds_per_root, self.effective_branches_per_decision) < 0.0:
            raise ValueError("tree-cost evidence must be nonnegative")


@dataclass(frozen=True)
class CandidateSelectionEvidence:
    candidate_id: str
    learning_gate_pass: bool
    domains: Mapping[str, DomainSelectionEvidence]

    def __post_init__(self) -> None:
        if set(self.domains) != set(DOMAINS):
            raise ValueError("candidate selection evidence requires both frozen domains")


def _mean(values: Sequence[float]) -> float:
    return float(sum(float(value) for value in values) / len(values))


def _validate_pairing(evidence: Mapping[str, CandidateSelectionEvidence]) -> None:
    for domain in DOMAINS:
        omission_sizes = {len(row.domains[domain].omission_samples) for row in evidence.values()}
        crossplay_sizes = {len(row.domains[domain].crossplay_samples) for row in evidence.values()}
        if len(omission_sizes) != 1 or len(crossplay_sizes) != 1:
            raise ValueError(f"paired sample identity/length mismatch in {domain}")


def _worst_domain(row: CandidateSelectionEvidence, metric: str) -> str:
    if metric == "omission":
        scored = [(_mean(row.domains[d].omission_samples), d) for d in DOMAINS]
        worst_value = max(value for value, _ in scored)
        return sorted(domain for value, domain in scored if value == worst_value)[0]
    if metric == "crossplay":
        scored = [(_mean(row.domains[d].crossplay_samples), d) for d in DOMAINS]
        worst_value = min(value for value, _ in scored)
        return sorted(domain for value, domain in scored if value == worst_value)[0]
    raise ValueError(metric)


def _pooled_bootstrap(diff: Sequence[float], *, metric: str, root_level: int, domain: str, a: str, b: str):
    return paired_bootstrap_mean_ci(
        diff,
        seed_fields=(
            "bootstrap",
            metric,
            int(root_level),
            str(domain),
            "POOLED",
            ordered_candidate_pair(a, b),
        ),
    )


def _rank_omission(survivors: list[str], evidence, root_level: int, trace: list[dict]) -> list[str]:
    absolute = {
        cid: max(_mean(evidence[cid].domains[d].omission_samples) for d in DOMAINS)
        for cid in survivors
    }
    best = min(survivors, key=lambda cid: (absolute[cid], cid))
    kept = [best]
    comparisons = []
    for cid in survivors:
        if cid == best:
            continue
        domain = _worst_domain(evidence[cid], "omission")
        diff = paired_difference(
            evidence[cid].domains[domain].omission_samples,
            evidence[best].domains[domain].omission_samples,
        )
        ci = _pooled_bootstrap(diff, metric="omission", root_level=root_level, domain=domain, a=cid, b=best)
        materially_worse = bool(ci["mean"] > MATERIAL_FLOOR and ci["ci_low"] > 0.0)
        comparisons.append({"challenger":cid,"best":best,"domain":domain,"difference":ci,"materially_worse":materially_worse})
        if not materially_worse:
            kept.append(cid)
    trace.append({"rank":2,"metric":"omission","absolute_scores":absolute,"best":best,"comparisons":comparisons,"survivors":sorted(kept)})
    return sorted(kept)


def _rank_crossplay(survivors: list[str], evidence, root_level: int, trace: list[dict]) -> list[str]:
    absolute = {
        cid: min(_mean(evidence[cid].domains[d].crossplay_samples) for d in DOMAINS)
        for cid in survivors
    }
    best = max(survivors, key=lambda cid: (absolute[cid], tuple(-ord(ch) for ch in cid)))
    # Re-select lexical-smallest on exact score tie without opaque string tricks.
    top = max(absolute.values())
    best = sorted(cid for cid in survivors if absolute[cid] == top)[0]
    kept = [best]
    comparisons = []
    for cid in survivors:
        if cid == best:
            continue
        domain = _worst_domain(evidence[cid], "crossplay")
        diff = paired_difference(
            evidence[best].domains[domain].crossplay_samples,
            evidence[cid].domains[domain].crossplay_samples,
        )
        ci = _pooled_bootstrap(diff, metric="crossplay", root_level=root_level, domain=domain, a=cid, b=best)
        materially_worse = bool(ci["mean"] > MATERIAL_FLOOR and ci["ci_low"] > 0.0)
        comparisons.append({"challenger":cid,"best":best,"domain":domain,"difference":ci,"materially_worse":materially_worse})
        if not materially_worse:
            kept.append(cid)
    trace.append({"rank":3,"metric":"crossplay","absolute_scores":absolute,"best":best,"comparisons":comparisons,"survivors":sorted(kept)})
    return sorted(kept)


def _rank_band(survivors: list[str], evidence, *, field: str, rank: int, trace: list[dict]) -> list[str]:
    scores = {
        cid: max(float(getattr(evidence[cid].domains[d], field)) for d in DOMAINS)
        for cid in survivors
    }
    best = min(scores.values())
    limit = 0.0 if best == 0.0 else best * 1.10
    kept = sorted(cid for cid in survivors if scores[cid] <= limit)
    trace.append({"rank":rank,"metric":field,"scores":scores,"best_score":best,"equivalence_limit":limit,"survivors":kept})
    return kept


def _rank_branches(survivors: list[str], evidence, trace: list[dict]) -> list[str]:
    scores = {
        cid: max(float(evidence[cid].domains[d].effective_branches_per_decision) for d in DOMAINS)
        for cid in survivors
    }
    best = min(scores.values())
    kept = sorted(cid for cid in survivors if scores[cid] == best)
    trace.append({"rank":6,"metric":"effective_branches_per_decision","scores":scores,"best_score":best,"survivors":kept})
    return kept


def prune_action_level(
    evidence: Mapping[str, CandidateSelectionEvidence],
    *,
    root_level: int,
    prior_eligible_ids: Sequence[str] | None = None,
) -> dict:
    if int(root_level) not in (160, 320, 640):
        raise ValueError("root_level must be 160, 320 or 640")
    if REFEREE not in evidence:
        raise ValueError("PF_DENSE_REFERENCE evidence is required")
    _validate_pairing(evidence)
    if not evidence[REFEREE].learning_gate_pass:
        return {"status":"BLOCKED","reason":"DENSE_REFEREE_LEARNING_GATE_FAILURE","root_level":int(root_level),"ready_for_tables":False}

    all_eligible = sorted(cid for cid in evidence if cid != REFEREE)
    if prior_eligible_ids is None:
        prior = set(all_eligible)
    else:
        prior = {str(cid) for cid in prior_eligible_ids}
        unknown = prior - set(all_eligible)
        if unknown:
            raise ValueError(f"prior eligible candidates missing from evidence: {sorted(unknown)}")
    survivors = sorted(cid for cid in all_eligible if cid in prior and evidence[cid].learning_gate_pass)
    trace: list[dict] = [{"rank":1,"metric":"learning_gates","prior_eligible":sorted(prior),"survivors":survivors}]
    if not survivors:
        return {"status":"FAIL","reason":"NO_CANDIDATE_PASSES_LEARNING_GATES","root_level":int(root_level),"trace":trace,"ready_for_tables":False}

    survivors = _rank_omission(survivors, evidence, int(root_level), trace)
    if len(survivors) > 1:
        survivors = _rank_crossplay(survivors, evidence, int(root_level), trace)
    if len(survivors) > 1:
        survivors = _rank_band(survivors, evidence, field="nodes_per_root", rank=4, trace=trace)
    if len(survivors) > 1:
        survivors = _rank_band(survivors, evidence, field="tree_seconds_per_root", rank=5, trace=trace)
    if len(survivors) > 1:
        survivors = _rank_branches(survivors, evidence, trace)

    selected = None
    next_level = None
    fallback_used = False
    if int(root_level) == 160:
        next_level = 320
    elif int(root_level) == 320:
        if len(survivors) == 1:
            selected = survivors[0]
        else:
            next_level = 640
    else:
        if len(survivors) == 1:
            selected = survivors[0]
        else:
            selected = CONTROL if CONTROL in survivors else sorted(survivors)[0]
            fallback_used = True
            trace.append({"rank":7,"metric":"conservative_fallback","selected":selected,"survivors_before_fallback":survivors})

    mandatory_execution = sorted(set(survivors) | {CONTROL, REFEREE}) if next_level else []
    control_only = sorted(cid for cid in mandatory_execution if cid not in survivors and cid != REFEREE)
    return {
        "status":"PASS_LEVEL",
        "root_level":int(root_level),
        "survivors":survivors,
        "selected_candidate":selected,
        "next_level":next_level,
        "mandatory_next_level_execution":mandatory_execution,
        "control_only_noneligible":control_only,
        "fallback_used":fallback_used,
        "trace":trace,
        "production_training_authorized":False,
        "ready_for_tables":False,
    }
