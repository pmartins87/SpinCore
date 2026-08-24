from __future__ import annotations

"""Phase2B12: nested IID conditional-chance expectation convergence.

Read-only diagnostic over the exact completed Phase2B6 behavior ensembles.  The
first 16 samples in every block deliberately reproduce the Phase2B11 IID16
control; samples 16..63 extend the exact same IID stream.  Raw Advantage targets
are prefix-averaged before diagnostic regret matching at K={8,16,32,64}.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import torch

import r7_5_arch_reset_v1plus_phase2b1_target_variance as b1
import r7_5_arch_reset_v1plus_phase2b10_private_public_chance_decomposition as b10
import r7_5_arch_reset_v1plus_phase2b11_factorized_chance_estimator as b11
from spincore.r7_5_action_cfr import regret_matching_policy
from spincore.r7_5_action_scenarios import action_scenario_cycle
from spincore.r7_5_representation_v3_stage_contract import TRAINING_SEEDS

SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B12_IID_CHANCE_EXPECTATION_CONVERGENCE_V1"
DOMAIN = "THREE_HANDED"
PHASE2B1_RESULT_SHA256 = "f95751afeb17fcd5844bfcb2971577b92a400750444e5dabe2f4ddb5718ba6ef"
PHASE2B6_RESULT_SHA256 = "33ec6ba89823dae632b7af935def17444379c96a28e59478c0b7c91f1ec3659a"
PHASE2B10_RESULT_SHA256 = "0295574c6133eb05866ecbdccf7e31efa4e6e8936dbd8bb7e375e166b27fe4dc"
PHASE2B11_RESULT_SHA256 = "1596023d39609ddfe5a6528a2e62d376c8e6bd29dde68d24a20a9b0ed782b1aa"
ANCHORS_PER_SCENARIO = 4
BLOCKS = 4
K_VALUES = (8, 16, 32, 64)
MAX_K = max(K_VALUES)
IID_NAMESPACE = 301  # exact Phase2B11 IID16 namespace; first 16 deals must reproduce B11
MAX_WORKERS = 30
REPRO_TOL = 1e-12
TV_ABS_GATE = 0.08
TV_REL_GATE = 0.25
K64_MAX_MEAN_TV = 0.24
SIGN_ABS_GATE = 0.05
SIGN_REL_GATE = 0.20
TAIL_THRESHOLD = 0.35
TAIL_ABS_GATE = 0.08
TAIL_REL_GATE = 0.20
DOMINANT_MISMATCH_TOLERANCE = 0.02
MONOTONE_TOLERANCE = 0.01
SLOW_ABS_IMPROVEMENT = 0.05

B11_REPRO = {
    "pooled_mean_tv": 0.33467186760867673,
    "pooled_sign": 0.2520833333333333,
    "pooled_target_mad": 0.006673636639562426,
    "pooled_dominant_mismatch": 0.3333333333333333,
    "seed_1342191342_mean_tv": 0.3171922054847577,
    "seed_1801739323_mean_tv": 0.3521515297325957,
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _summary(values: Sequence[float]) -> dict:
    rows = np.asarray([float(v) for v in values], dtype=np.float64)
    if rows.size == 0:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(rows.size),
        "mean": float(rows.mean()),
        "p50": float(np.quantile(rows, 0.50, method="linear")),
        "p95": float(np.quantile(rows, 0.95, method="linear")),
        "max": float(rows.max()),
    }


def _policy_tv(left: Sequence[float], right: Sequence[float]) -> float:
    return float(0.5 * sum(abs(float(a) - float(b)) for a, b in zip(left, right)))


def _mean_targets(targets: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not targets:
        raise ValueError("Phase2B12 cannot average an empty target set")
    width = len(targets[0])
    if width != 10 or any(len(row) != width for row in targets):
        raise ValueError("Phase2B12 target width drift")
    return tuple(
        float(sum(float(row[index]) for row in targets) / len(targets))
        for index in range(width)
    )


def _worker_init(repo_root: str, solver_path: str, behavior_seed: int, behavior_states: list[dict]) -> None:
    b10._worker_init(repo_root, solver_path, int(behavior_seed), behavior_states)


def _worker_task(task: dict) -> dict:
    if b10._WORKER_SOLVER is None or b10._WORKER_ACTION_SPEC is None:
        raise RuntimeError("Phase2B12 worker not initialized")
    scenarios = action_scenario_cycle(DOMAIN)
    scenario_index = int(task["scenario_index"])
    anchor_index = int(task["anchor_index"])
    block = int(task["block"])
    episode = scenarios[scenario_index]
    expected = {
        "observation_sha256": str(task["observation_sha256"]),
        "actor": int(task["actor"]),
        "legal": tuple(int(x) for x in task["legal"]),
        "legal_mask": tuple(int(x) for x in task["legal_mask"]),
    }

    anchor = b10._WORKER_SOLVER.create(episode, int(task["anchor_deck_seed"]))
    try:
        observation, actor, legal, mask = b1._root_identity(anchor, b10._WORKER_ACTION_SPEC)
        if (
            hashlib.sha256(observation).hexdigest() != expected["observation_sha256"]
            or actor != expected["actor"]
            or legal != expected["legal"]
            or mask != expected["legal_mask"]
        ):
            raise RuntimeError("Phase2B12 stored Phase2B1 anchor identity drift")
        snapshot = anchor.deal_snapshot()
    finally:
        anchor.close()
    if snapshot.visible_board_count != 0:
        raise RuntimeError("Phase2B12 anchor must be an initial preflop root")

    deals = b11._iid_deals(
        snapshot,
        actor,
        scenario_index=scenario_index,
        anchor_index=anchor_index,
        block=block,
        count=MAX_K,
        namespace=IID_NAMESPACE,
    )
    if len(deals) != MAX_K:
        raise AssertionError("Phase2B12 IID stream length drift")

    fixed_traversal = b11._traversal_seed(scenario_index, anchor_index)
    targets = []
    nodes = 0
    started = time.perf_counter()
    estimators = {}
    k_cursor = 0
    for index, deal in enumerate(deals, start=1):
        target, node_count = b10._one_target(episode, deal, fixed_traversal, expected)
        targets.append(target)
        nodes += int(node_count)
        if k_cursor < len(K_VALUES) and index == K_VALUES[k_cursor]:
            estimators[str(index)] = [float(x) for x in _mean_targets(targets)]
            k_cursor += 1
    if tuple(sorted(int(k) for k in estimators)) != K_VALUES:
        raise AssertionError("Phase2B12 missing nested K estimator")

    return {
        "source_behavior_seed": int(b10._WORKER_BEHAVIOR_SEED),
        "scenario_index": scenario_index,
        "anchor_index": anchor_index,
        "block": block,
        "anchor_deck_seed": int(task["anchor_deck_seed"]),
        "legal_mask": list(expected["legal_mask"]),
        "estimators": estimators,
        "nodes": int(nodes),
        "seconds": float(time.perf_counter() - started),
    }


def _tasks(collision_groups: Sequence[dict]) -> list[dict]:
    if len(collision_groups) != 15:
        raise RuntimeError("Phase2B12 requires exactly 15 Phase2B1 collision groups")
    tasks = []
    for group in collision_groups:
        seeds = [int(x) for x in group.get("deck_seeds") or []]
        if len(seeds) < ANCHORS_PER_SCENARIO:
            raise RuntimeError("Phase2B12 collision group lacks four frozen anchors")
        for anchor_index in range(ANCHORS_PER_SCENARIO):
            for block in range(BLOCKS):
                tasks.append({
                    "scenario_index": int(group["scenario_index"]),
                    "anchor_index": int(anchor_index),
                    "block": int(block),
                    "anchor_deck_seed": int(seeds[anchor_index]),
                    "observation_sha256": str(group["observation_sha256"]),
                    "actor": int(group["actor"]),
                    "legal": tuple(int(x) for x in group["legal"]),
                    "legal_mask": tuple(int(x) for x in group["legal_mask"]),
                })
    return tasks


def _run_behavior_seed(
    repo_root: Path,
    solver_path: Path,
    behavior_seed: int,
    behavior_states: list[dict],
    collision_groups: Sequence[dict],
    workers: int,
) -> list[dict]:
    tasks = _tasks(collision_groups)
    results = []
    with ProcessPoolExecutor(
        max_workers=int(workers),
        initializer=_worker_init,
        initargs=(str(repo_root), str(solver_path), int(behavior_seed), behavior_states),
    ) as pool:
        future_map = {pool.submit(_worker_task, task): task for task in tasks}
        for future in as_completed(future_map):
            task = future_map[future]
            row = future.result()
            results.append(row)
            print(
                f"[Phase2B12 target] behavior={behavior_seed} scenario={task['scenario_index']:02d} "
                f"anchor={task['anchor_index']} block={task['block']} seconds={row['seconds']:.2f}",
                flush=True,
            )
    results.sort(
        key=lambda row: (
            int(row["source_behavior_seed"]),
            int(row["scenario_index"]),
            int(row["anchor_index"]),
            int(row["block"]),
        )
    )
    return results


def _pair_metric(left: Sequence[float], right: Sequence[float], legal_mask_row: Sequence[int]) -> dict:
    legal = tuple(index for index, enabled in enumerate(legal_mask_row) if int(enabled))
    if not legal:
        raise ValueError("Phase2B12 empty legal mask")
    mad = float(sum(abs(float(left[a]) - float(right[a])) for a in legal) / len(legal))
    sign = float(sum((float(left[a]) > 0.0) != (float(right[a]) > 0.0) for a in legal) / len(legal))
    lp = regret_matching_policy(left, legal)
    rp = regret_matching_policy(right, legal)
    tv = _policy_tv(lp, rp)
    ldom = max(legal, key=lambda a: float(lp[a]))
    rdom = max(legal, key=lambda a: float(rp[a]))
    return {
        "target_mean_abs_diff": mad,
        "legal_sign_disagreement_fraction": sign,
        "regret_matching_policy_tv": tv,
        "dominant_legal_action_mismatch": int(ldom != rdom),
        "tv_ge_035": int(tv >= TAIL_THRESHOLD),
    }


def _summaries(task_rows: Sequence[dict]) -> tuple[list[dict], dict, dict]:
    index = {
        (
            int(row["source_behavior_seed"]),
            int(row["scenario_index"]),
            int(row["anchor_index"]),
            int(row["block"]),
        ): row
        for row in task_rows
    }
    pair_rows = []
    for seed in map(int, TRAINING_SEEDS):
        for scenario_index in range(15):
            for anchor_index in range(ANCHORS_PER_SCENARIO):
                for pair_start in (0, 2):
                    left_row = index[(seed, scenario_index, anchor_index, pair_start)]
                    right_row = index[(seed, scenario_index, anchor_index, pair_start + 1)]
                    if tuple(left_row["legal_mask"]) != tuple(right_row["legal_mask"]):
                        raise RuntimeError("Phase2B12 paired block legal-mask drift")
                    for k in K_VALUES:
                        metric = _pair_metric(
                            left_row["estimators"][str(k)],
                            right_row["estimators"][str(k)],
                            left_row["legal_mask"],
                        )
                        pair_rows.append({
                            "source_behavior_seed": seed,
                            "scenario_index": scenario_index,
                            "anchor_index": anchor_index,
                            "pair_start": pair_start,
                            "k": int(k),
                            **metric,
                        })

    def summarize(rows: Sequence[dict]) -> dict:
        return {
            "pair_count": len(rows),
            "target_mean_abs_diff": _summary([row["target_mean_abs_diff"] for row in rows]),
            "legal_sign_disagreement_fraction": _summary([row["legal_sign_disagreement_fraction"] for row in rows]),
            "regret_matching_policy_tv": _summary([row["regret_matching_policy_tv"] for row in rows]),
            "dominant_legal_action_mismatch_rate": float(
                sum(int(row["dominant_legal_action_mismatch"]) for row in rows) / len(rows)
            ) if rows else None,
            "tail_rate_tv_ge_035": float(sum(int(row["tv_ge_035"]) for row in rows) / len(rows)) if rows else None,
        }

    by_seed = {}
    for seed in map(int, TRAINING_SEEDS):
        by_seed[str(seed)] = {}
        for k in K_VALUES:
            rows = [
                row for row in pair_rows
                if int(row["source_behavior_seed"]) == seed and int(row["k"]) == k
            ]
            by_seed[str(seed)][str(k)] = summarize(rows)

    pooled = {}
    for k in K_VALUES:
        pooled[str(k)] = summarize([row for row in pair_rows if int(row["k"]) == k])
    return pair_rows, by_seed, pooled


def _close(actual: float, expected: float, name: str) -> None:
    if abs(float(actual) - float(expected)) > REPRO_TOL:
        raise RuntimeError(
            f"Phase2B12 B11 IID16 reproduction failed for {name}: {actual!r} != {expected!r}"
        )


def _reproduction_gate(by_seed: dict, pooled: dict) -> dict:
    k16 = pooled["16"]
    checks = {
        "pooled_mean_tv": float(k16["regret_matching_policy_tv"]["mean"]),
        "pooled_sign": float(k16["legal_sign_disagreement_fraction"]["mean"]),
        "pooled_target_mad": float(k16["target_mean_abs_diff"]["mean"]),
        "pooled_dominant_mismatch": float(k16["dominant_legal_action_mismatch_rate"]),
        "seed_1342191342_mean_tv": float(by_seed["1342191342"]["16"]["regret_matching_policy_tv"]["mean"]),
        "seed_1801739323_mean_tv": float(by_seed["1801739323"]["16"]["regret_matching_policy_tv"]["mean"]),
    }
    for name, value in checks.items():
        _close(value, B11_REPRO[name], name)
    return {"pass": True, "tolerance": REPRO_TOL, "observed": checks, "expected": dict(B11_REPRO)}


def _decision(by_seed: dict, pooled: dict) -> dict:
    k16 = pooled["16"]
    k32 = pooled["32"]
    k64 = pooled["64"]

    control_tv = float(k16["regret_matching_policy_tv"]["mean"])
    candidate_tv = float(k64["regret_matching_policy_tv"]["mean"])
    tv_abs = control_tv - candidate_tv
    tv_rel = tv_abs / control_tv if control_tv > 0.0 else 0.0

    control_sign = float(k16["legal_sign_disagreement_fraction"]["mean"])
    candidate_sign = float(k64["legal_sign_disagreement_fraction"]["mean"])
    sign_abs = control_sign - candidate_sign
    sign_rel = sign_abs / control_sign if control_sign > 0.0 else 0.0

    control_tail = float(k16["tail_rate_tv_ge_035"])
    candidate_tail = float(k64["tail_rate_tv_ge_035"])
    tail_abs = control_tail - candidate_tail
    tail_rel = tail_abs / control_tail if control_tail > 0.0 else 0.0

    tv_material = bool(tv_abs >= TV_ABS_GATE and tv_rel >= TV_REL_GATE)
    absolute_residual = bool(candidate_tv <= K64_MAX_MEAN_TV)
    sign_material = bool(sign_abs >= SIGN_ABS_GATE or sign_rel >= SIGN_REL_GATE)
    tail_material = bool(tail_abs >= TAIL_ABS_GATE or tail_rel >= TAIL_REL_GATE)
    both_seed = all(
        float(by_seed[str(seed)]["64"]["regret_matching_policy_tv"]["mean"])
        < float(by_seed[str(seed)]["16"]["regret_matching_policy_tv"]["mean"])
        for seed in map(int, TRAINING_SEEDS)
    )
    monotone = bool(
        float(k32["regret_matching_policy_tv"]["mean"])
        <= float(k16["regret_matching_policy_tv"]["mean"]) + MONOTONE_TOLERANCE
        and float(k64["regret_matching_policy_tv"]["mean"])
        <= float(k32["regret_matching_policy_tv"]["mean"]) + MONOTONE_TOLERANCE
    )
    dom_ok = bool(
        float(k64["dominant_legal_action_mismatch_rate"])
        <= float(k16["dominant_legal_action_mismatch_rate"]) + DOMINANT_MISMATCH_TOLERANCE
    )

    full_pass = bool(
        tv_material
        and absolute_residual
        and sign_material
        and tail_material
        and both_seed
        and monotone
        and dom_ok
    )
    if full_pass:
        status = "IID_CHANCE_EXPECTATION_CONVERGES_MATERIALLY"
        route = "PRECOMMIT_SMALL_MULTI_CHANCE_TARGET_TRAINING_PILOT_WITH_EQUAL_COMPUTE_CONTROL"
        small_pilot = True
    elif tv_abs >= SLOW_ABS_IMPROVEMENT and both_seed:
        status = "IID_CHANCE_EXPECTATION_CONVERGES_SLOWLY"
        route = "QUANTIFY_COMPUTE_FRONTIER_OR_REPRESENTATION_SUPPORT_BEFORE_TRAINING"
        small_pilot = False
    else:
        status = "IID_CHANCE_EXPECTATION_PLATEAUS_OR_UNRESOLVED"
        route = "REASSESS_REPRESENTATION_AND_REGRET_SIGN_SENSITIVITY_NO_TRAINING"
        small_pilot = False

    return {
        "classification": status,
        "k_mean_tv": {
            str(k): float(pooled[str(k)]["regret_matching_policy_tv"]["mean"])
            for k in K_VALUES
        },
        "k_tail_rate_tv_ge_035": {
            str(k): float(pooled[str(k)]["tail_rate_tv_ge_035"])
            for k in K_VALUES
        },
        "k_sign_disagreement": {
            str(k): float(pooled[str(k)]["legal_sign_disagreement_fraction"]["mean"])
            for k in K_VALUES
        },
        "k_target_mad": {
            str(k): float(pooled[str(k)]["target_mean_abs_diff"]["mean"])
            for k in K_VALUES
        },
        "k_dominant_mismatch": {
            str(k): float(pooled[str(k)]["dominant_legal_action_mismatch_rate"])
            for k in K_VALUES
        },
        "k64_vs_k16": {
            "tv_absolute_improvement": float(tv_abs),
            "tv_relative_improvement": float(tv_rel),
            "sign_absolute_improvement": float(sign_abs),
            "sign_relative_improvement": float(sign_rel),
            "tail_absolute_improvement": float(tail_abs),
            "tail_relative_improvement": float(tail_rel),
        },
        "tv_materiality_pass": tv_material,
        "absolute_k64_residual_pass": absolute_residual,
        "sign_materiality_pass": sign_material,
        "tail_materiality_pass": tail_material,
        "both_source_behavior_seeds_directionally_improve": both_seed,
        "monotone_convergence_guardrail_pass": monotone,
        "dominant_action_mismatch_non_degradation_pass": dom_ok,
        "screen_pass": full_pass,
        "next_route": route,
        "small_causal_training_pilot_precommit_allowed": small_pilot,
        "training_pilot_authorized": False,
        "architecture_winner_selected": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }


def _validate_result(path: Path, expected_sha: str, expected_status: str | None = None) -> dict:
    if _sha256(path) != expected_sha:
        raise RuntimeError(f"Phase2B12 prerequisite SHA mismatch for {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if expected_status is not None and payload.get("status") != expected_status:
        raise RuntimeError(f"Phase2B12 prerequisite status mismatch for {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="R7.5 Phase2B12 nested IID chance expectation convergence")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--phase2b1-result", type=Path, required=True)
    parser.add_argument("--phase2b6-root", type=Path, required=True)
    parser.add_argument("--phase2b6-result", type=Path, required=True)
    parser.add_argument("--phase2b10-result", type=Path, required=True)
    parser.add_argument("--phase2b11-result", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    solver_path = args.solver.resolve()
    b1_result = _validate_result(args.phase2b1_result.resolve(), PHASE2B1_RESULT_SHA256)
    _validate_result(
        args.phase2b6_result.resolve(),
        PHASE2B6_RESULT_SHA256,
        "PREFLOP_DAMPING_CAUSAL_EFFECT_SUPPORTED_BUT_STILL_UNSTABLE",
    )
    _validate_result(args.phase2b10_result.resolve(), PHASE2B10_RESULT_SHA256, "MIXED_PRIVATE_PUBLIC_CHANCE")
    _validate_result(args.phase2b11_result.resolve(), PHASE2B11_RESULT_SHA256, "FACTORIZED_CHANCE_ESTIMATOR_SCREEN_FAIL")
    if b1_result.get("schema") != b1.SCHEMA:
        raise RuntimeError("Phase2B12 wrong Phase2B1 source schema")
    collision_groups = list(b1_result.get("collision_groups") or [])
    workers = max(1, min(int(args.workers), MAX_WORKERS, os.cpu_count() or MAX_WORKERS))
    torch.set_num_threads(1)

    b6_root = args.phase2b6_root.resolve()
    states_by_seed = {}
    checkpoint_identity = []
    for seed in map(int, TRAINING_SEEDS):
        checkpoint = b6_root / f"seed_{seed}" / "resume_checkpoint.pt"
        states = b10._load_b6_behavior_states(checkpoint, seed)
        states_by_seed[seed] = states
        checkpoint_identity.append(
            {
                "training_seed": seed,
                "path": str(checkpoint),
                "sha256": _sha256(checkpoint),
                "behavior_members": len(states),
            }
        )

    started = time.perf_counter()
    task_rows = []
    seconds_by_seed = {}
    for seed in map(int, TRAINING_SEEDS):
        local = time.perf_counter()
        print(
            f"[Phase2B12] behavior seed {seed}: 240 tasks / 15360 root traversals with {workers} workers...",
            flush=True,
        )
        rows = _run_behavior_seed(
            repo_root,
            solver_path,
            seed,
            states_by_seed[seed],
            collision_groups,
            workers,
        )
        task_rows.extend(rows)
        seconds_by_seed[str(seed)] = float(time.perf_counter() - local)

    pair_rows, by_seed, pooled = _summaries(task_rows)
    reproduction = _reproduction_gate(by_seed, pooled)
    decision = _decision(by_seed, pooled)
    result = {
        "schema": SCHEMA,
        "status": decision["classification"],
        "representation": b10.REPRESENTATION,
        "domain": DOMAIN,
        "source_behavior": "EXACT_COMPLETED_PHASE2B6_WITH_25_PERCENT_PREFLOP_CONTINUATION_FLOOR",
        "training_seeds": [int(seed) for seed in TRAINING_SEEDS],
        "anchors_per_scenario": ANCHORS_PER_SCENARIO,
        "blocks": BLOCKS,
        "k_values": list(K_VALUES),
        "iid_namespace": IID_NAMESPACE,
        "nested_prefix_stream": True,
        "total_root_target_traversals": 30720,
        "worker_processes": workers,
        "torch_threads_per_worker": 1,
        "pair_metric_row_count": len(pair_rows),
        "b11_iid16_reproduction": reproduction,
        "by_source_behavior_seed": by_seed,
        "pooled": pooled,
        "decision": decision,
        "runtime_seconds_by_source_behavior_seed": seconds_by_seed,
        "runtime_seconds_total": float(time.perf_counter() - started),
        "frozen_inputs": {
            "phase2b1_result_sha256": PHASE2B1_RESULT_SHA256,
            "phase2b6_result_sha256": PHASE2B6_RESULT_SHA256,
            "phase2b10_result_sha256": PHASE2B10_RESULT_SHA256,
            "phase2b11_result_sha256": PHASE2B11_RESULT_SHA256,
            "phase2b6_checkpoints": checkpoint_identity,
        },
        "guardrails": [
            "K16 is the exact Phase2B11 IID16 prefix and must reproduce B11 summaries within 1e-12.",
            "K8/K16/K32/K64 are nested prefixes of the same IID conditional-chance stream.",
            "Traversal RNG is fixed per scenario/anchor; only legal hidden/future chance is integrated.",
            "Raw target vectors are averaged before diagnostic regret matching.",
            "No model fit, optimizer step, reservoir insertion, Strategy collection, AveragePolicy fit or checkpoint mutation occurs.",
            "A screen PASS allows only a separately precommitted small causal training pilot with equal-compute control.",
        ],
        "architecture_winner_selected": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": result["status"],
                "k_mean_tv": decision["k_mean_tv"],
                "k_tail_rate_tv_ge_035": decision["k_tail_rate_tv_ge_035"],
                "k64_vs_k16": decision["k64_vs_k16"],
                "screen_pass": decision["screen_pass"],
                "next_route": decision["next_route"],
                "runtime_seconds_total": result["runtime_seconds_total"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
