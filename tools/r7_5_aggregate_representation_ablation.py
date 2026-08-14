from __future__ import annotations

import argparse
import json
import math
import statistics
from itertools import combinations
from pathlib import Path

import torch

SCHEMA = "SPINCORE_R7_5_3_REPRESENTATION_ABLATION_RESULT_V1"
FIT_SCHEMA = "SPINCORE_R7_5_3_CANDIDATE_FIT_V1"
PRECOMMIT_SCHEMA = "SPINCORE_R7_5_3_REPRESENTATION_ABLATION_PRECOMMIT_V1"
AGGREGATION_SCHEMA = "SPINCORE_R7_5_3_AGGREGATION_IMPLEMENTATION_FREEZE_V1"


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _load_fit(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != FIT_SCHEMA:
        raise ValueError(f"wrong fit schema in {path}")
    report = dict(payload.get("report") or {})
    if report.get("schema") != FIT_SCHEMA:
        raise ValueError(f"missing fit report in {path}")
    identities = [bytes(value) for value in payload.get("policy_heldout_identity") or []]
    predictions = payload.get("policy_heldout_predictions") or []
    if len(identities) != len(predictions):
        raise ValueError(f"heldout identity/prediction length mismatch in {path}")
    return {
        "path": str(path),
        "report": report,
        "identities": identities,
        "predictions": torch.tensor(predictions, dtype=torch.float32),
    }


def _cross_fit(entries: list[dict], expected_seeds: list[int]) -> dict[str, float]:
    by_seed = {int(entry["report"]["fit_seed"]): entry for entry in entries}
    if set(by_seed) != set(expected_seeds):
        raise ValueError("fit-seed set differs from frozen precommit")
    ordered = [by_seed[seed] for seed in expected_seeds]
    identities = ordered[0]["identities"]
    for entry in ordered[1:]:
        if entry["identities"] != identities:
            raise ValueError("cross-fit heldout identities differ across fit seeds")
    if not identities:
        raise ValueError("cross-fit heldout set is empty")

    all_tv: list[torch.Tensor] = []
    pair_rows = []
    for left, right in combinations(ordered, 2):
        a = left["predictions"]
        b = right["predictions"]
        if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 6:
            raise ValueError("cross-fit prediction tensor shape mismatch")
        tv = 0.5 * torch.abs(a - b).sum(dim=1)
        all_tv.append(tv)
        pair_rows.append(
            {
                "fit_seed_a": int(left["report"]["fit_seed"]),
                "fit_seed_b": int(right["report"]["fit_seed"]),
                "mean_tv": float(tv.mean()),
                "p95_tv": float(torch.quantile(tv, torch.tensor(0.95))),
                "max_tv": float(tv.max()),
            }
        )
    merged = torch.cat(all_tv)
    return {
        "observation_count_per_fit_pair": len(identities),
        "pair_count": len(pair_rows),
        "aggregate_observation_pair_count": int(merged.numel()),
        "mean_tv": float(merged.mean()),
        "p50_tv": float(torch.quantile(merged, torch.tensor(0.50))),
        "p95_tv": float(torch.quantile(merged, torch.tensor(0.95))),
        "max_tv": float(merged.max()),
        "pairs": pair_rows,
    }


def _mean(values) -> float:
    values = [float(value) for value in values]
    if not values or not all(_finite(value) for value in values):
        raise ValueError("non-finite or empty metric set")
    return float(sum(values) / len(values))


def _candidate_summary(
    *,
    candidate: dict,
    domains: list[str],
    fit_seeds: list[int],
    grouped: dict[tuple[str, str], list[dict]],
    gates: dict,
    aggregation_freeze: dict,
) -> dict:
    candidate_id = str(candidate["id"])
    domain_rows = {}
    absolute_failures: list[str] = []
    expected_parameter_count = int(candidate["parameter_count"])

    for domain in domains:
        entries = grouped.get((candidate_id, domain), [])
        if len(entries) != len(fit_seeds):
            raise ValueError(
                f"candidate/domain requires {len(fit_seeds)} fits: {candidate_id}/{domain}"
            )
        by_seed = {int(entry["report"]["fit_seed"]): entry for entry in entries}
        if set(by_seed) != set(fit_seeds):
            raise ValueError(f"fit seeds mismatch for {candidate_id}/{domain}")
        ordered = [by_seed[seed] for seed in fit_seeds]

        reference_corpus = ordered[0]["report"]["corpus"]
        reference_counts = ordered[0]["report"]["counts"]
        for entry in ordered:
            report = entry["report"]
            if report["corpus"] != reference_corpus or report["counts"] != reference_counts:
                raise ValueError(f"corpus/split drift across fit seeds for {candidate_id}/{domain}")
            if int(report["parameter_count"]) != expected_parameter_count:
                raise ValueError(f"parameter-count drift for {candidate_id}/{domain}")
            if not bool(report["absolute_gates"]["advantage_pass"]):
                absolute_failures.append(
                    f"{domain}/fit_seed={report['fit_seed']}/advantage_nrmse"
                )
            if not bool(report["absolute_gates"]["policy_pass"]):
                absolute_failures.append(
                    f"{domain}/fit_seed={report['fit_seed']}/policy_tv"
                )

        cross = _cross_fit(ordered, fit_seeds)
        cross_pass = bool(
            float(cross["mean_tv"]) <= float(gates["cross_fit_mean_tv_max"])
            and float(cross["p95_tv"]) <= float(gates["cross_fit_p95_tv_max"])
        )
        if not cross_pass:
            absolute_failures.append(f"{domain}/cross_fit")

        sentinel_values = []
        inference_seconds_per_sample = []
        fit_rows = []
        for entry in ordered:
            report = entry["report"]
            sentinel_count = int(report["heldout_advantage"]["eligible_sentinel_count"])
            sentinel_metric = float(report["heldout_advantage"]["sentinel_macro_weighted_nrmse"])
            if sentinel_count <= 0 or not _finite(sentinel_metric):
                raise ValueError(f"no eligible semantic sentinels for {candidate_id}/{domain}")
            sentinel_values.append(sentinel_metric)
            heldout_count = int(report["counts"]["strategy_heldout"])
            if heldout_count <= 0:
                raise ValueError("empty policy heldout count")
            inference_seconds_per_sample.append(
                float(report["heldout_policy"]["inference_seconds"]) / heldout_count
            )
            fit_rows.append(
                {
                    "fit_seed": int(report["fit_seed"]),
                    "advantage_weighted_nrmse": float(
                        report["heldout_advantage"]["weighted_nrmse"]
                    ),
                    "policy_weighted_mean_tv": float(
                        report["heldout_policy"]["weighted_mean_tv"]
                    ),
                    "sentinel_macro_weighted_nrmse": sentinel_metric,
                    "eligible_sentinel_count": sentinel_count,
                    "policy_inference_seconds_per_sample": inference_seconds_per_sample[-1],
                    "fit_gate_pass": bool(report["absolute_gates"]["fit_pass"]),
                    "peak_rss_kib": int(report["peak_rss_kib"]),
                }
            )

        domain_rows[domain] = {
            "fits": fit_rows,
            "mean_advantage_weighted_nrmse": _mean(
                row["advantage_weighted_nrmse"] for row in fit_rows
            ),
            "mean_policy_weighted_mean_tv": _mean(
                row["policy_weighted_mean_tv"] for row in fit_rows
            ),
            "mean_sentinel_macro_weighted_nrmse": _mean(sentinel_values),
            "median_policy_inference_seconds_per_sample": float(
                statistics.median(inference_seconds_per_sample)
            ),
            "cross_fit": cross,
            "cross_fit_gate_pass": cross_pass,
            "corpus": reference_corpus,
            "counts": reference_counts,
        }

    summary = {
        "candidate": candidate_id,
        "flop_candidate": candidate.get("flop_candidate"),
        "parameter_count": expected_parameter_count,
        "serialized_observation_bytes": 126 if candidate_id == "C0_V1_FROZEN_CONTROL" else 830,
        "active_flop_tokens": candidate.get("active_flop_tokens"),
        "domains": domain_rows,
        "absolute_failures": absolute_failures,
        "absolute_gate_pass": not absolute_failures,
    }
    summary["selection_metrics"] = {
        "worst_domain_advantage_nrmse": max(
            row["mean_advantage_weighted_nrmse"] for row in domain_rows.values()
        ),
        "worst_domain_policy_tv": max(
            row["mean_policy_weighted_mean_tv"] for row in domain_rows.values()
        ),
        "worst_domain_sentinel_macro_nrmse": max(
            row["mean_sentinel_macro_weighted_nrmse"] for row in domain_rows.values()
        ),
        "worst_domain_cross_fit_p95_tv": max(
            row["cross_fit"]["p95_tv"] for row in domain_rows.values()
        ),
        "worst_domain_policy_inference_seconds_per_sample": max(
            row["median_policy_inference_seconds_per_sample"] for row in domain_rows.values()
        ),
    }
    return summary


def _band_step(
    alive: list[str],
    summaries: dict[str, dict],
    *,
    metric: str,
    absolute_band: float | None = None,
    relative_band: float | None = None,
) -> tuple[list[str], dict]:
    values = {candidate: float(summaries[candidate]["selection_metrics"][metric]) for candidate in alive}
    if not values or not all(_finite(value) for value in values.values()):
        raise ValueError(f"non-finite selection metric: {metric}")
    best = min(values.values())
    if absolute_band is not None:
        threshold = best + float(absolute_band)
    elif relative_band is not None:
        threshold = best * (1.0 + float(relative_band))
    else:
        threshold = best
    survivors = sorted(candidate for candidate, value in values.items() if value <= threshold)
    return survivors, {
        "metric": metric,
        "values": values,
        "best": best,
        "threshold": threshold,
        "survivors": survivors,
    }


def _exact_min_step(
    alive: list[str],
    summaries: dict[str, dict],
    *,
    field: str,
) -> tuple[list[str], dict]:
    values = {candidate: int(summaries[candidate][field]) for candidate in alive}
    best = min(values.values())
    survivors = sorted(candidate for candidate, value in values.items() if value == best)
    return survivors, {
        "metric": field,
        "values": values,
        "best": best,
        "survivors": survivors,
    }


def apply_frozen_selection(summaries: dict[str, dict]) -> tuple[str | None, list[dict]]:
    trace: list[dict] = []
    alive = sorted(
        candidate for candidate, summary in summaries.items() if summary["absolute_gate_pass"]
    )
    trace.append(
        {
            "rank": 1,
            "metric": "absolute_gates",
            "survivors": list(alive),
            "discarded": sorted(set(summaries) - set(alive)),
        }
    )
    if not alive:
        return None, trace

    band_steps = [
        (2, "worst_domain_advantage_nrmse", 0.015, None),
        (3, "worst_domain_policy_tv", 0.01, None),
        (4, "worst_domain_sentinel_macro_nrmse", 0.02, None),
        (5, "worst_domain_cross_fit_p95_tv", 0.02, None),
        (6, "worst_domain_policy_inference_seconds_per_sample", None, 0.05),
    ]
    for rank, metric, absolute_band, relative_band in band_steps:
        if len(alive) <= 1:
            break
        alive, row = _band_step(
            alive,
            summaries,
            metric=metric,
            absolute_band=absolute_band,
            relative_band=relative_band,
        )
        row["rank"] = rank
        trace.append(row)

    for rank, field in ((7, "parameter_count"), (8, "serialized_observation_bytes")):
        if len(alive) <= 1:
            break
        alive, row = _exact_min_step(alive, summaries, field=field)
        row["rank"] = rank
        trace.append(row)

    if len(alive) > 1:
        v2_alive = [
            candidate
            for candidate in alive
            if summaries[candidate].get("active_flop_tokens") is not None
        ]
        if len(v2_alive) == len(alive):
            alive, row = _exact_min_step(alive, summaries, field="active_flop_tokens")
            row["rank"] = 9
            trace.append(row)
        else:
            trace.append(
                {
                    "rank": 9,
                    "metric": "active_flop_tokens",
                    "survivors": list(alive),
                    "note": "not applied across mixed V1/V2 survivors",
                }
            )

    if len(alive) > 1:
        winner = "C0_V1_FROZEN_CONTROL" if "C0_V1_FROZEN_CONTROL" in alive else sorted(alive)[0]
        trace.append(
            {
                "rank": 10,
                "metric": "deterministic_final_fallback",
                "survivors_before": list(alive),
                "winner": winner,
            }
        )
        alive = [winner]

    return alive[0], trace


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate frozen R7.5.3 paired representation ablation")
    parser.add_argument("--fit", type=Path, action="append", required=True)
    parser.add_argument("--precommit", type=Path, required=True)
    parser.add_argument("--aggregation-freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    precommit = json.loads(args.precommit.read_text(encoding="utf-8"))
    aggregation_freeze = json.loads(args.aggregation_freeze.read_text(encoding="utf-8"))
    if precommit.get("schema") != PRECOMMIT_SCHEMA:
        raise ValueError("wrong R7.5.3 precommit schema")
    if aggregation_freeze.get("schema") != AGGREGATION_SCHEMA:
        raise ValueError("wrong R7.5.3 aggregation freeze schema")
    if bool(precommit.get("ready_for_tables")) or bool(aggregation_freeze.get("ready_for_tables")):
        raise ValueError("R7.5.3 freeze must not authorize table use")

    candidates = [dict(row) for row in precommit["candidates"] if row.get("eligible_to_win")]
    candidate_ids = [str(row["id"]) for row in candidates]
    domains = [str(value) for value in precommit["domains"]]
    fit_seeds = [int(value) for value in precommit["seed_derivation"]["fit_initialization_seeds"]]
    raw_gates = dict(precommit["unchanged_absolute_gates"])
    gates = {
        "advantage_weighted_nrmse_max": float(raw_gates["advantage_weighted_nrmse_max"]),
        "policy_weighted_mean_tv_max": float(raw_gates["policy_weighted_mean_tv_max"]),
        "cross_fit_mean_tv_max": float(raw_gates["cross_fit_mean_tv_max"]),
        "cross_fit_p95_tv_max": float(raw_gates["cross_fit_p95_tv_max"]),
    }

    grouped: dict[tuple[str, str], list[dict]] = {}
    seen_keys = set()
    for path in args.fit:
        entry = _load_fit(path)
        report = entry["report"]
        key = (str(report["candidate"]), str(report["domain"]), int(report["fit_seed"]))
        if key in seen_keys:
            raise ValueError(f"duplicate fit result: {key}")
        seen_keys.add(key)
        if key[0] not in candidate_ids or key[1] not in domains or key[2] not in fit_seeds:
            raise ValueError(f"unexpected candidate/domain/seed fit: {key}")
        grouped.setdefault((key[0], key[1]), []).append(entry)

    expected_fit_count = len(candidate_ids) * len(domains) * len(fit_seeds)
    if len(seen_keys) != expected_fit_count:
        raise ValueError(f"expected {expected_fit_count} fit outputs, found {len(seen_keys)}")

    summaries = {}
    for candidate in candidates:
        summary = _candidate_summary(
            candidate=candidate,
            domains=domains,
            fit_seeds=fit_seeds,
            grouped=grouped,
            gates=gates,
            aggregation_freeze=aggregation_freeze,
        )
        summaries[summary["candidate"]] = summary

    winner, selection_trace = apply_frozen_selection(summaries)
    passed = winner is not None
    payload = {
        "schema": SCHEMA,
        "precommit_schema": precommit["schema"],
        "precommit_semantic_anchor_sha": precommit["semantic_anchor_sha"],
        "aggregation_freeze_schema": aggregation_freeze["schema"],
        "candidate_count": len(candidate_ids),
        "domain_count": len(domains),
        "fit_seed_count": len(fit_seeds),
        "fit_output_count": len(seen_keys),
        "unchanged_absolute_gates": gates,
        "candidates": summaries,
        "selection_trace": selection_trace,
        "selected_candidate": winner,
        "r7_5_3_representation_ablation_pass": bool(passed),
        "production_representation_selected": bool(passed),
        "production_training_authorized": False,
        "r7_5_4_pending": True,
        "r7_5_5_pending": True,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
