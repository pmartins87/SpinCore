from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path


TARGETS = {
    "policy_mixture_size4_256": "validation/R7_3_PARTIAL_EXACT_POLICY_MIXTURE_PAIRED_SIZE4_256.json",
    "policy_mixture_size4_320": "validation/R7_3_PARTIAL_EXACT_POLICY_MIXTURE_COMPOUNDING_320.json",
}


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _history(path: str):
    text = _git("log", "--format=%H", "-n", "2", "--", path)
    return [x for x in text.splitlines() if x]


def _load_at(commit: str, path: str):
    return json.loads(_git("show", f"{commit}:{path}"))


def _core(payload: dict):
    cross = payload.get("cross_seed", {})
    per_seed = payload.get("per_seed", [])
    return {
        "schema": payload.get("schema"),
        "ensemble_size": payload.get("ensemble_size"),
        "iterations": payload.get("iterations"),
        "roots_per_iteration": payload.get("roots_per_iteration"),
        "roots_per_seed": payload.get("roots_per_seed"),
        "exact_opponent_levels": payload.get("exact_opponent_levels"),
        "cross_seed": {
            "mean_tv": cross.get("mean_tv"),
            "p50_tv": cross.get("p50_tv"),
            "p95_tv": cross.get("p95_tv"),
            "max_tv": cross.get("max_tv"),
        },
        "per_seed_fit_pass": payload.get("per_seed_fit_pass"),
        "cross_seed_pass": payload.get("cross_seed_pass"),
        "r7_3_pass": payload.get("r7_3_pass"),
        "per_seed": [
            {
                "algorithm_seed": row.get("algorithm_seed"),
                "roots": row.get("roots"),
                "nodes": row.get("nodes"),
                "advantage_seen": row.get("advantage_seen"),
                "strategy_seen": row.get("strategy_seen"),
                "final_fit": row.get("final_fit"),
            }
            for row in per_seed
        ],
    }


def _num_delta(a, b):
    if a is None or b is None:
        return None
    try:
        return abs(float(a) - float(b))
    except Exception:
        return None


def _compare(old: dict, new: dict):
    a = _core(old)
    b = _core(new)
    cross_deltas = {
        key: _num_delta(a["cross_seed"].get(key), b["cross_seed"].get(key))
        for key in ("mean_tv", "p50_tv", "p95_tv", "max_tv")
    }
    counter_equal = []
    fit_deltas = []
    for left, right in zip(a["per_seed"], b["per_seed"]):
        counter_equal.append({
            "algorithm_seed": left.get("algorithm_seed"),
            "same_seed": left.get("algorithm_seed") == right.get("algorithm_seed"),
            "roots_equal": left.get("roots") == right.get("roots"),
            "nodes_equal": left.get("nodes") == right.get("nodes"),
            "advantage_seen_equal": left.get("advantage_seen") == right.get("advantage_seen"),
            "strategy_seen_equal": left.get("strategy_seen") == right.get("strategy_seen"),
        })
        lf = left.get("final_fit") or {}
        rf = right.get("final_fit") or {}
        common = sorted(set(lf) & set(rf))
        fit_deltas.append({
            "algorithm_seed": left.get("algorithm_seed"),
            "numeric_abs_delta": {
                key: _num_delta(lf.get(key), rf.get(key))
                for key in common
                if isinstance(lf.get(key), (int, float)) and not isinstance(lf.get(key), bool)
            },
            "boolean_equal": {
                key: lf.get(key) == rf.get(key)
                for key in common
                if isinstance(lf.get(key), bool)
            },
        })
    finite_deltas = [x for x in cross_deltas.values() if x is not None and math.isfinite(x)]
    max_cross_delta = max(finite_deltas) if finite_deltas else math.inf
    exact_counters = all(all(v for k, v in row.items() if k.endswith("_equal")) for row in counter_equal)
    exact_structure = all(
        a.get(k) == b.get(k)
        for k in ("schema", "ensemble_size", "iterations", "roots_per_iteration", "roots_per_seed", "exact_opponent_levels", "per_seed_fit_pass", "cross_seed_pass", "r7_3_pass")
    )
    return {
        "cross_seed_abs_delta": cross_deltas,
        "max_cross_seed_abs_delta": float(max_cross_delta),
        "counter_equality": counter_equal,
        "fit_deltas": fit_deltas,
        "exact_structure_equal": bool(exact_structure),
        "exact_sample_and_node_counters_equal": bool(exact_counters),
        "cross_seed_equal_within_1e_9": bool(max_cross_delta <= 1e-9),
        "fresh_run_reproducible_core": bool(exact_structure and exact_counters and max_cross_delta <= 1e-9),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare latest two physical R7.3 evidence commits for fresh-run reproducibility")
    ap.add_argument("--out", type=Path, default=Path("validation/R7_3_FRESH_RUN_REPRODUCIBILITY.json"))
    args = ap.parse_args()

    targets = {}
    complete = True
    for label, path in TARGETS.items():
        commits = _history(path)
        if len(commits) < 2:
            targets[label] = {"path": path, "status": "WAITING_FOR_SECOND_EVIDENCE_COMMIT", "commits": commits}
            complete = False
            continue
        newest, previous = commits[0], commits[1]
        old = _load_at(previous, path)
        new = _load_at(newest, path)
        if old.get("runner_failed_before_report") or new.get("runner_failed_before_report"):
            targets[label] = {
                "path": path,
                "status": "EVIDENCE_CONTAINS_RUNNER_FAILURE_MARKER",
                "older_commit": previous,
                "newer_commit": newest,
            }
            complete = False
            continue
        targets[label] = {
            "path": path,
            "status": "COMPARED",
            "older_commit": previous,
            "newer_commit": newest,
            "comparison": _compare(old, new),
        }

    payload = {
        "schema": "SPINCORE_R7_3_FRESH_RUN_REPRODUCIBILITY_V1",
        "complete": bool(complete),
        "targets": targets,
        "all_core_reproducible": bool(
            complete and all(x["comparison"]["fresh_run_reproducible_core"] for x in targets.values())
        ),
        "interpretation_note": (
            "This is a fresh-process/fresh-run determinism check, not checkpoint/resume recertification. "
            "Generated timestamps and wall-clock durations are intentionally excluded. Core comparison "
            "requires exact structural/sample/node counters and cross-seed metrics within 1e-9. A new "
            "production behavior semantic still requires continuous-vs-stop/restore/continue testing."
        ),
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if complete else 3


if __name__ == "__main__":
    raise SystemExit(main())
