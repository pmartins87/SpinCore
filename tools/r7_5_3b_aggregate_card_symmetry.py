from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import torch

import r7_5_aggregate_representation_ablation as oldagg

SCHEMA = "SPINCORE_R7_5_3B_CARD_SYMMETRY_RESULT_V1"
FIT_SCHEMA = "SPINCORE_R7_5_3B_CARD_SYMMETRY_FIT_V1"
PRECOMMIT_SCHEMA = "SPINCORE_R7_5_3B_CARD_SYMMETRY_PRECOMMIT_V1"
VARIANTS = ("S0_V1_FROZEN_CONTROL", "S1_V1_CARD_SYMMETRY_CANON")


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _load_fit(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != FIT_SCHEMA:
        raise ValueError(f"wrong 3B fit schema in {path}")
    report = dict(payload.get("report") or {})
    if report.get("schema") != FIT_SCHEMA:
        raise ValueError(f"missing 3B fit report in {path}")
    variant = str(report.get("variant"))
    if variant not in VARIANTS:
        raise ValueError(f"unknown 3B variant in {path}: {variant}")
    identities = [bytes(value) for value in payload.get("policy_heldout_identity") or []]
    predictions = payload.get("policy_heldout_predictions") or []
    if len(identities) != len(predictions) or not identities:
        raise ValueError(f"heldout identity/prediction mismatch in {path}")
    return {
        "path": str(path),
        "variant": variant,
        "report": report,
        "identities": identities,
        "predictions": torch.tensor(predictions, dtype=torch.float32),
    }


def _mean(values) -> float:
    values = [float(value) for value in values]
    if not values or not all(_finite(value) for value in values):
        raise ValueError("non-finite or empty metric set")
    return float(sum(values) / len(values))


def _summarize_variant(entries: list[dict], pre: dict, variant: str) -> dict:
    domains = list(pre["domains"])
    fit_seeds = [int(value) for value in pre["fit_initialization_seeds"]]
    gates = dict(pre["unchanged_absolute_gates"])
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        if entry["variant"] == variant:
            by_domain[str(entry["report"]["domain"])].append(entry)

    domain_rows = {}
    failures: list[str] = []
    for domain in domains:
        rows = by_domain.get(domain, [])
        if len(rows) != len(fit_seeds):
            raise ValueError(f"{variant}/{domain} expected {len(fit_seeds)} fits, got {len(rows)}")
        by_seed = {int(row["report"]["fit_seed"]): row for row in rows}
        if set(by_seed) != set(fit_seeds):
            raise ValueError(f"fit-seed set mismatch for {variant}/{domain}")
        ordered = [by_seed[seed] for seed in fit_seeds]

        source_corpus = ordered[0]["report"]["source_corpus"]
        counts = ordered[0]["report"]["counts"]
        fit_rows = []
        for entry in ordered:
            report = entry["report"]
            if report["source_corpus"] != source_corpus or report["counts"] != counts:
                raise ValueError(f"source corpus drift across seeds for {variant}/{domain}")
            if int(report["parameter_count"]) != 152438:
                raise ValueError(f"parameter-count drift for {variant}/{domain}")
            if int(report["serialized_observation_bytes"]) != 126:
                raise ValueError(f"wire-width drift for {variant}/{domain}")

            adv = float(report["heldout_advantage"]["weighted_nrmse"])
            pol = float(report["heldout_policy"]["weighted_mean_tv"])
            sentinel = float(report["heldout_advantage"]["sentinel_macro_weighted_nrmse"])
            sentinel_count = int(report["heldout_advantage"]["eligible_sentinel_count"])
            if not _finite(adv) or not _finite(pol) or not _finite(sentinel) or sentinel_count <= 0:
                raise ValueError(f"invalid heldout metric for {variant}/{domain}")

            adv_pass = adv <= float(gates["advantage_weighted_nrmse_max"])
            pol_pass = pol <= float(gates["policy_weighted_mean_tv_max"])
            if not adv_pass:
                failures.append(f"{domain}/fit_seed={report['fit_seed']}/advantage")
            if not pol_pass:
                failures.append(f"{domain}/fit_seed={report['fit_seed']}/policy")

            heldout_count = int(report["counts"]["strategy_heldout"])
            inference_sps = float(report["heldout_policy"]["inference_seconds"]) / heldout_count
            if not _finite(inference_sps) or inference_sps <= 0.0:
                raise ValueError("invalid inference timing")
            fit_rows.append(
                {
                    "fit_seed": int(report["fit_seed"]),
                    "advantage_weighted_nrmse": adv,
                    "policy_weighted_mean_tv": pol,
                    "sentinel_macro_weighted_nrmse": sentinel,
                    "eligible_sentinel_count": sentinel_count,
                    "model_inference_seconds_per_sample": inference_sps,
                    "fit_gate_pass": bool(adv_pass and pol_pass),
                    "changed_advantage_observations": int(
                        report["transformation"]["changed_advantage_observations"]
                    ),
                    "changed_strategy_observations": int(
                        report["transformation"]["changed_strategy_observations"]
                    ),
                }
            )

        cross = oldagg._cross_fit(ordered, fit_seeds)
        cross_pass = bool(
            float(cross["mean_tv"]) <= float(gates["cross_fit_mean_tv_max"])
            and float(cross["p95_tv"]) <= float(gates["cross_fit_p95_tv_max"])
        )
        if not cross_pass:
            failures.append(f"{domain}/cross_fit")

        domain_rows[domain] = {
            "fits": fit_rows,
            "source_corpus": source_corpus,
            "counts": counts,
            "mean_advantage_weighted_nrmse": _mean(
                row["advantage_weighted_nrmse"] for row in fit_rows
            ),
            "mean_policy_weighted_mean_tv": _mean(
                row["policy_weighted_mean_tv"] for row in fit_rows
            ),
            "mean_sentinel_macro_weighted_nrmse": _mean(
                row["sentinel_macro_weighted_nrmse"] for row in fit_rows
            ),
            "median_model_inference_seconds_per_sample": float(
                statistics.median(row["model_inference_seconds_per_sample"] for row in fit_rows)
            ),
            "cross_fit": cross,
            "cross_fit_gate_pass": cross_pass,
        }

    selection_metrics = {
        "worst_domain_mean_advantage_weighted_nrmse": max(
            row["mean_advantage_weighted_nrmse"] for row in domain_rows.values()
        ),
        "worst_domain_mean_policy_weighted_mean_tv": max(
            row["mean_policy_weighted_mean_tv"] for row in domain_rows.values()
        ),
        "worst_domain_mean_sentinel_macro_nrmse": max(
            row["mean_sentinel_macro_weighted_nrmse"] for row in domain_rows.values()
        ),
        "worst_domain_cross_fit_p95_tv": max(
            float(row["cross_fit"]["p95_tv"]) for row in domain_rows.values()
        ),
        "worst_domain_model_inference_seconds_per_sample": max(
            row["median_model_inference_seconds_per_sample"] for row in domain_rows.values()
        ),
    }
    return {
        "variant": variant,
        "parameter_count": 152438,
        "serialized_observation_bytes": 126,
        "domains": domain_rows,
        "absolute_failures": failures,
        "absolute_gate_pass": not failures,
        "selection_metrics": selection_metrics,
    }


def _assert_exact_pairing(summaries: dict[str, dict], pre: dict) -> None:
    for domain in pre["domains"]:
        left = summaries[VARIANTS[0]]["domains"][domain]
        right = summaries[VARIANTS[1]]["domains"][domain]
        if left["source_corpus"] != right["source_corpus"] or left["counts"] != right["counts"]:
            raise ValueError(f"S0/S1 source corpus mismatch in {domain}")


def _select(summaries: dict[str, dict], pre: dict) -> tuple[str | None, list[dict]]:
    s0 = summaries[VARIANTS[0]]
    s1 = summaries[VARIANTS[1]]
    trace: list[dict] = []
    pass0 = bool(s0["absolute_gate_pass"])
    pass1 = bool(s1["absolute_gate_pass"])
    trace.append({"step": "absolute_gates", "S0_pass": pass0, "S1_pass": pass1})
    if not pass0 and not pass1:
        return None, trace
    if pass0 and not pass1:
        return VARIANTS[0], trace
    if pass1 and not pass0:
        return VARIANTS[1], trace

    bands = dict(pre["noninferiority_bands_reused_from_R7_5_3"])
    comparisons = [
        ("worst_domain_mean_advantage_weighted_nrmse", "worst_domain_mean_advantage_weighted_nrmse_absolute", "absolute"),
        ("worst_domain_mean_policy_weighted_mean_tv", "worst_domain_mean_policy_weighted_mean_tv_absolute", "absolute"),
        ("worst_domain_mean_sentinel_macro_nrmse", "worst_domain_mean_sentinel_macro_nrmse_absolute", "absolute"),
        ("worst_domain_cross_fit_p95_tv", "worst_domain_cross_fit_p95_tv_absolute", "absolute"),
        ("worst_domain_model_inference_seconds_per_sample", "model_inference_seconds_per_sample_relative", "relative"),
    ]
    noninferior = True
    for metric, band_key, mode in comparisons:
        value0 = float(s0["selection_metrics"][metric])
        value1 = float(s1["selection_metrics"][metric])
        band = float(bands[band_key])
        threshold = value0 + band if mode == "absolute" else value0 * (1.0 + band)
        passed = bool(value1 <= threshold)
        trace.append({
            "step": "S1_noninferiority",
            "metric": metric,
            "S0": value0,
            "S1": value1,
            "mode": mode,
            "band": band,
            "threshold": threshold,
            "pass": passed,
        })
        noninferior = noninferior and passed
    return (VARIANTS[1] if noninferior else VARIANTS[0]), trace


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate R7.5.3B card-symmetry gate")
    parser.add_argument("--fit", type=Path, action="append", required=True)
    parser.add_argument("--precommit", type=Path, required=True)
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--source-corpus-run-id", type=int, default=31767822186)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    pre = json.loads(args.precommit.read_text(encoding="utf-8"))
    if pre.get("schema") != PRECOMMIT_SCHEMA:
        raise ValueError("wrong R7.5.3B precommit schema")
    if pre.get("status") != "FROZEN_BEFORE_CARD_SYMMETRY_COMPARATIVE_OUTPUTS":
        raise ValueError("R7.5.3B precommit is not frozen")
    if int(args.source_corpus_run_id) != int(pre["paired_corpus_source"]["workflow_run_id"]):
        raise ValueError("source corpus run differs from frozen precommit")

    entries = [_load_fit(path) for path in args.fit]
    expected_count = len(VARIANTS) * len(pre["domains"]) * len(pre["fit_initialization_seeds"])
    if len(entries) != expected_count:
        raise ValueError(f"expected {expected_count} fits, got {len(entries)}")

    summaries = {variant: _summarize_variant(entries, pre, variant) for variant in VARIANTS}
    _assert_exact_pairing(summaries, pre)
    winner, trace = _select(summaries, pre)
    status = "PASS" if winner is not None else "BLOCKED"

    result = {
        "schema": SCHEMA,
        "status": status,
        "winner_id": winner,
        "execution_sha": str(args.execution_sha),
        "source_corpus_run_id": int(args.source_corpus_run_id),
        "precommit": str(args.precommit),
        "summaries": summaries,
        "selection_trace": trace,
        "representation_decision": {
            "existing_R7_5_4A_evidence_representation_consistent": winner == "S0_V1_FROZEN_CONTROL",
            "S1_requires_cpp_integration_runtime_certification_before_production": winner == "S1_V1_CARD_SYMMETRY_CANON",
            "S1_requires_fresh_action_abstraction_validation": winner == "S1_V1_CARD_SYMMETRY_CANON",
        },
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if winner is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
