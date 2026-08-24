from __future__ import annotations

"""Phase2B16: exact-rejection posterior continuation chance screen.

Final estimator-level diagnostic after Phase2B15.  Opponent private cards are
sampled exactly from the action-history posterior by rejection sampling from the
uniform private-card prior with acceptance probability equal to the frozen
behavior-policy likelihood of the already-observed preflop path.  Future boards
are sampled only after acceptance.  No network fitting or reservoir mutation is
performed.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Sequence

import numpy as np

import r7_5_arch_reset_v1plus_phase2b10_private_public_chance_decomposition as b10
import r7_5_arch_reset_v1plus_phase2b11_factorized_chance_estimator as b11
import r7_5_arch_reset_v1plus_phase2b13_root_iid64_target_training as b13
import r7_5_arch_reset_v1plus_phase2b15_posterior_weighted_continuation_chance as b15
import r7_5_arch_reset_v1plus_phase2b15_posterior_weighted_continuation_chance_runtimefix as b15fix

from spincore.r7_5_action_cfr import regret_matching_policy, validate_policy
from spincore.r7_5_action_scenarios import action_scenario_cycle
from spincore.r7_5_representation_v3_referee_states import effective_pf0
from spincore.r7_5_representation_v3_stage_contract import TRAINING_SEEDS, EVALUATION_SEEDS
from spincore.solver_v3 import neural_bytes_v3

SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B16_EXACT_REJECTION_POSTERIOR_CONTINUATION_V1"
PARTIAL_SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B16_PARTIAL_V1"
DOMAIN = b15.DOMAIN
REGIONS = b15.REGIONS
K = 64
BLOCKS = 2
MAX_WORKERS = 30
MAX_PROPOSALS_PER_TASK = 50000
B15_RESULT_SHA256 = "0e4f0a5bf2d48fb7f48b2763f8a65e3093d879aa50729f5d8a80d28fa9578f6a"
B15_STATUS = "POSTERIOR_WEIGHTING_MATERIAL_BUT_STABILITY_NOT_SUPPORTED"

# Frozen B16 gates.
SNIS_TV_ABS_IMPROVEMENT_MIN = 0.05
SNIS_TV_REL_IMPROVEMENT_MIN = 0.20
DIRECT_TV_MAX = 0.24
SNIS_SIGN_ABS_IMPROVEMENT_MIN = 0.03
SNIS_SIGN_REL_IMPROVEMENT_MIN = 0.15
DIRECT_TAIL_MAX = 0.28
DIRECT_DOMINANT_MISMATCH_MAX = 0.28
REGION_MAX_DEGRADE_VS_SNIS = 0.01
MASK64 = (1 << 64) - 1


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _summary(values: Sequence[float]) -> dict:
    arr = np.asarray([float(x) for x in values], dtype=np.float64)
    if not arr.size:
        return {"count": 0, "mean": None, "p10": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p10": float(np.quantile(arr, 0.10, method="linear")),
        "p50": float(np.quantile(arr, 0.50, method="linear")),
        "p95": float(np.quantile(arr, 0.95, method="linear")),
        "max": float(arr.max()),
    }


def _mix64(*parts: int) -> int:
    x = 0x2B160FACADE00001
    for raw in parts:
        y = int(raw) & MASK64
        x ^= (y + 0x9E3779B97F4A7C15 + ((x << 6) & MASK64) + (x >> 2)) & MASK64
        x ^= x >> 30
        x = (x * 0xBF58476D1CE4E5B9) & MASK64
        x ^= x >> 27
        x = (x * 0x94D049BB133111EB) & MASK64
        x ^= x >> 31
    return x & MASK64


def _proposal_seeds(evaluation_seed: int, state_index: int, block: int, proposal_index: int) -> tuple[int, int, int]:
    key = (int(evaluation_seed), int(state_index), int(block), int(proposal_index))
    return (
        _mix64(0x1601, *key),  # private cards
        _mix64(0x1602, *key),  # probe filler board
        _mix64(0x1603, *key),  # acceptance uniform
    )


def _accepted_board_seed(evaluation_seed: int, state_index: int, block: int, accepted_index: int, proposal_index: int) -> int:
    return _mix64(0x1604, int(evaluation_seed), int(state_index), int(block), int(accepted_index), int(proposal_index))


def _open_unit(seed: int) -> float:
    rng = random.Random(int(seed))
    return float((rng.getrandbits(53) + 0.5) / float(1 << 53))


def _accept_log_likelihood(log_likelihood: float, u: float) -> bool:
    if not 0.0 < float(u) < 1.0:
        raise ValueError("Phase2B16 rejection uniform must be strictly inside (0,1)")
    if not math.isfinite(float(log_likelihood)):
        return False
    if float(log_likelihood) > 1e-12:
        raise RuntimeError("Phase2B16 behavior path likelihood exceeded one")
    return bool(math.log(float(u)) <= float(log_likelihood))


def _log_likelihood(task: dict, deal) -> float:
    if b10._WORKER_SOLVER is None or b10._WORKER_COLLECTOR is None or b10._WORKER_ACTION_SPEC is None:
        raise RuntimeError("Phase2B16 worker not initialized")
    episode = action_scenario_cycle(DOMAIN)[int(task["scenario_index"])]
    state = b10._WORKER_SOLVER.create_with_deal(episode, deal.holes, deal.board)
    collector = b10._WORKER_COLLECTOR
    log_likelihood = 0.0
    try:
        for action in task["action_path"]:
            if state.terminal:
                raise RuntimeError("Phase2B16 proposal path reaches terminal early")
            observation = neural_bytes_v3(state)
            active_mask, legal, _exact = effective_pf0(state, b10._WORKER_ACTION_SPEC)
            if int(action) not in legal:
                raise RuntimeError("Phase2B16 proposal path action became illegal")
            probabilities = validate_policy(collector.policy(state, observation, legal), legal)
            probability = float(probabilities[int(action)])
            if probability <= 0.0:
                log_likelihood = -math.inf
            elif math.isfinite(log_likelihood):
                log_likelihood += math.log(probability)
            state.apply_universal(active_mask, int(action))
        if state.terminal:
            raise RuntimeError("Phase2B16 proposal continuation unexpectedly terminal")
        observation = neural_bytes_v3(state)
        active_mask, legal, _exact = effective_pf0(state, b10._WORKER_ACTION_SPEC)
        if observation != bytes(task["observation"]):
            raise RuntimeError("Phase2B16 proposal changed target infoset observation")
        if int(state.actor) != int(task["actor"]):
            raise RuntimeError("Phase2B16 proposal changed target actor")
        if int(active_mask) != int(task["active_mask"]) or tuple(legal) != tuple(task["legal_slots"]):
            raise RuntimeError("Phase2B16 proposal changed target legal identity")
    finally:
        state.close()
    return float(log_likelihood)


def _mean_targets(targets: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if len(targets) != K or any(len(row) != 10 for row in targets):
        raise RuntimeError("Phase2B16 requires exactly K ten-action targets")
    return tuple(float(sum(float(row[i]) for row in targets) / K) for i in range(10))


def _worker_task(task: dict) -> dict:
    snapshot = b15fix._canonical_snapshot(task)
    actor = int(task["actor"])
    block = int(task["block"])
    traversal_seed = b15._traversal_seed(int(task["evaluation_seed"]), int(task["state_index"]))
    accepted_targets = []
    accepted_log_likelihoods = []
    proposals = 0
    nodes = 0
    started = time.perf_counter()

    while len(accepted_targets) < K and proposals < MAX_PROPOSALS_PER_TASK:
        proposal_index = int(proposals)
        private_seed, probe_board_seed, accept_seed = _proposal_seeds(
            int(task["evaluation_seed"]), int(task["state_index"]), block, proposal_index
        )
        probe_deal = b11._deal_from_factors(snapshot, actor, private_seed, probe_board_seed)
        log_likelihood = _log_likelihood(task, probe_deal)
        u = _open_unit(accept_seed)
        proposals += 1
        if not _accept_log_likelihood(log_likelihood, u):
            continue

        accepted_index = len(accepted_targets)
        board_seed = _accepted_board_seed(
            int(task["evaluation_seed"]), int(task["state_index"]), block,
            accepted_index, proposal_index,
        )
        accepted_deal = b11._deal_from_factors(snapshot, actor, private_seed, board_seed)
        target, verified_log_likelihood, node_count = b15._variant_likelihood_and_target(
            task, accepted_deal, traversal_seed
        )
        if math.isfinite(log_likelihood) != math.isfinite(verified_log_likelihood):
            raise RuntimeError("Phase2B16 accepted likelihood reproducibility drift")
        if math.isfinite(log_likelihood) and abs(float(log_likelihood) - float(verified_log_likelihood)) > 1e-12:
            raise RuntimeError("Phase2B16 accepted likelihood changed with future board")
        accepted_targets.append(tuple(float(x) for x in target))
        accepted_log_likelihoods.append(float(log_likelihood))
        nodes += int(node_count)

    cap_hit = len(accepted_targets) != K
    if cap_hit:
        return {
            "schema": PARTIAL_SCHEMA,
            "execution_sha": str(task["execution_sha"]),
            "behavior_seed": int(task["behavior_seed"]),
            "evaluation_seed": int(task["evaluation_seed"]),
            "state_index": int(task["state_index"]),
            "scenario_index": int(task["scenario_index"]),
            "region": str(task["region"]),
            "block": block,
            "actor": actor,
            "observation_sha256": str(task["observation_sha256"]),
            "k": K,
            "accepted": len(accepted_targets),
            "proposals": int(proposals),
            "proposal_cap_hit": True,
            "acceptance_rate": float(len(accepted_targets) / proposals) if proposals else 0.0,
            "seconds": float(time.perf_counter() - started),
        }

    probs = [math.exp(x) for x in accepted_log_likelihoods]
    return {
        "schema": PARTIAL_SCHEMA,
        "execution_sha": str(task["execution_sha"]),
        "behavior_seed": int(task["behavior_seed"]),
        "evaluation_seed": int(task["evaluation_seed"]),
        "state_index": int(task["state_index"]),
        "scenario_index": int(task["scenario_index"]),
        "region": str(task["region"]),
        "block": block,
        "k": K,
        "actor": actor,
        "action_path_length": len(task["action_path"]),
        "legal_slots": [int(x) for x in task["legal_slots"]],
        "observation_sha256": str(task["observation_sha256"]),
        "exact_posterior_target": [float(x) for x in _mean_targets(accepted_targets)],
        "accepted": K,
        "proposals": int(proposals),
        "proposal_cap_hit": False,
        "acceptance_rate": float(K / proposals),
        "accepted_likelihood": _summary(probs),
        "target_nodes": int(nodes),
        "target_traversals": K,
        "seconds": float(time.perf_counter() - started),
    }


def _partial_path(root: Path, task: dict) -> Path:
    return (
        root / "partials" / f"behavior_{int(task['behavior_seed'])}"
        / f"eval_{int(task['evaluation_seed'])}"
        / f"state_{int(task['state_index']):04d}_block_{int(task['block'])}.json"
    )


def _valid_partial(payload: dict, task: dict) -> bool:
    return bool(
        payload.get("schema") == PARTIAL_SCHEMA
        and payload.get("execution_sha") == task["execution_sha"]
        and int(payload.get("behavior_seed", -1)) == int(task["behavior_seed"])
        and int(payload.get("evaluation_seed", -1)) == int(task["evaluation_seed"])
        and int(payload.get("state_index", -1)) == int(task["state_index"])
        and int(payload.get("block", -1)) == int(task["block"])
        and int(payload.get("k", -1)) == K
        and payload.get("observation_sha256") == task["observation_sha256"]
    )


def _run_behavior_seed(*, repo_root: Path, solver_path: Path, output_root: Path, execution_sha: str,
                       behavior_seed: int, behavior_states: list[dict], anchors: Sequence[dict], workers: int) -> list[dict]:
    tasks, rows = [], []
    for anchor in anchors:
        for block in range(BLOCKS):
            task = dict(anchor)
            task.update({"block": int(block), "behavior_seed": int(behavior_seed), "execution_sha": str(execution_sha)})
            path = _partial_path(output_root, task)
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if _valid_partial(payload, task):
                    rows.append(payload)
                    continue
            tasks.append(task)
    if tasks:
        with ProcessPoolExecutor(
            max_workers=min(int(workers), len(tasks)),
            initializer=b10._worker_init,
            initargs=(str(repo_root), str(solver_path), int(behavior_seed), behavior_states),
        ) as pool:
            fmap = {pool.submit(_worker_task, task): task for task in tasks}
            for future in as_completed(fmap):
                task = fmap[future]
                row = future.result()
                _atomic_json(row, _partial_path(output_root, task))
                rows.append(row)
                print(
                    f"[Phase2B16 task] behavior={behavior_seed} eval={task['evaluation_seed']} "
                    f"state={task['state_index']} {task['region']} block={task['block']} "
                    f"accepted={row['accepted']}/{K} proposals={row['proposals']} "
                    f"rate={row['acceptance_rate']:.5f} seconds={row['seconds']:.2f}",
                    flush=True,
                )
    expected = len(anchors) * BLOCKS
    if len(rows) != expected:
        raise RuntimeError(f"Phase2B16 behavior task coverage drift: {len(rows)} != {expected}")
    rows.sort(key=lambda r: (int(r["evaluation_seed"]), int(r["state_index"]), int(r["block"])))
    return rows


def _load_b15_partials(root: Path, anchors: Sequence[dict]) -> list[dict]:
    rows = []
    for behavior_seed in map(int, TRAINING_SEEDS):
        for anchor in anchors:
            for block in range(BLOCKS):
                task = dict(anchor)
                task.update({"behavior_seed": behavior_seed, "block": block})
                path = b15._partial_path(root, task)
                if not path.is_file():
                    raise RuntimeError(f"Phase2B16 missing successful Phase2B15 runtimefix partial: {path}")
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("runtimefix_schema") != b15fix.FIX_SCHEMA:
                    raise RuntimeError("Phase2B16 requires Phase2B15 runtimefix partials")
                required = bool(
                    payload.get("schema") == b15.PARTIAL_SCHEMA
                    and int(payload.get("behavior_seed", -1)) == behavior_seed
                    and int(payload.get("evaluation_seed", -1)) == int(anchor["evaluation_seed"])
                    and int(payload.get("state_index", -1)) == int(anchor["state_index"])
                    and int(payload.get("block", -1)) == block
                    and int(payload.get("k", -1)) == b15.K
                    and payload.get("observation_sha256") == anchor["observation_sha256"]
                )
                if not required:
                    raise RuntimeError("Phase2B16 Phase2B15 partial identity drift")
                rows.append(payload)
    return rows


def _verify_b15_baseline(rows: Sequence[dict], result: dict) -> tuple[list[dict], dict]:
    pairs = b15._pair_rows(rows)
    pooled = b15._aggregate(pairs)
    checks = {
        "unweighted_tv": (pooled["unweighted_tv"]["mean"], result["pooled"]["unweighted_tv"]["mean"]),
        "posterior_tv": (pooled["posterior_tv"]["mean"], result["pooled"]["posterior_tv"]["mean"]),
        "unweighted_sign": (pooled["unweighted_sign_disagreement_mean"], result["pooled"]["unweighted_sign_disagreement_mean"]),
        "posterior_sign": (pooled["posterior_sign_disagreement_mean"], result["pooled"]["posterior_sign_disagreement_mean"]),
        "unweighted_tail": (pooled["unweighted_tail_rate_tv_ge_035"], result["pooled"]["unweighted_tail_rate_tv_ge_035"]),
        "posterior_tail": (pooled["posterior_tail_rate_tv_ge_035"], result["pooled"]["posterior_tail_rate_tv_ge_035"]),
    }
    for name, (actual, expected) in checks.items():
        if abs(float(actual) - float(expected)) > 1e-12:
            raise RuntimeError(f"Phase2B16 Phase2B15 partial aggregate reproduction drift {name}: {actual} != {expected}")
    return pairs, pooled


def _pair_direct(rows: Sequence[dict]) -> list[dict]:
    groups = defaultdict(dict)
    for row in rows:
        key = (int(row["behavior_seed"]), int(row["evaluation_seed"]), int(row["state_index"]))
        groups[key][int(row["block"])] = row
    out = []
    for key, blocks in sorted(groups.items()):
        if set(blocks) != {0, 1}:
            raise RuntimeError(f"Phase2B16 missing paired block for {key}")
        left, right = blocks[0], blocks[1]
        if bool(left.get("proposal_cap_hit")) or bool(right.get("proposal_cap_hit")):
            out.append({
                "behavior_seed": key[0], "evaluation_seed": key[1], "state_index": key[2],
                "scenario_index": int(left["scenario_index"]), "region": str(left["region"]),
                "proposal_cap_hit": True,
            })
            continue
        legal = tuple(int(x) for x in left["legal_slots"])
        lt = tuple(float(x) for x in left["exact_posterior_target"])
        rt = tuple(float(x) for x in right["exact_posterior_target"])
        lp = regret_matching_policy(lt, legal)
        rp = regret_matching_policy(rt, legal)
        tv = b15._policy_tv(lp, rp)
        sign = b15._sign_disagreement(lt, rt, legal)
        mad = b15._legal_target_mad(lt, rt, legal)
        dom = b15._dominant_mismatch(lt, rt, legal)
        out.append({
            "behavior_seed": key[0], "evaluation_seed": key[1], "state_index": key[2],
            "scenario_index": int(left["scenario_index"]), "region": str(left["region"]),
            "proposal_cap_hit": False, "direct_tv": float(tv), "direct_sign_disagreement": float(sign),
            "direct_target_mad": float(mad), "direct_dominant_mismatch": int(dom),
            "acceptance_rate_mean": 0.5 * (float(left["acceptance_rate"]) + float(right["acceptance_rate"])),
            "proposals_max": max(int(left["proposals"]), int(right["proposals"])),
        })
    return out


def _aggregate_direct(pairs: Sequence[dict]) -> dict:
    valid = [r for r in pairs if not bool(r.get("proposal_cap_hit"))]
    cap_hits = len(pairs) - len(valid)
    if not valid:
        return {"count": len(pairs), "valid_count": 0, "proposal_cap_hits": cap_hits}
    tv = [float(r["direct_tv"]) for r in valid]
    sign = [float(r["direct_sign_disagreement"]) for r in valid]
    mad = [float(r["direct_target_mad"]) for r in valid]
    dom = [float(r["direct_dominant_mismatch"]) for r in valid]
    rates = [float(r["acceptance_rate_mean"]) for r in valid]
    props = [float(r["proposals_max"]) for r in valid]
    return {
        "count": len(pairs), "valid_count": len(valid), "proposal_cap_hits": cap_hits,
        "direct_tv": _summary(tv),
        "direct_sign_disagreement_mean": float(sum(sign) / len(sign)),
        "direct_target_mad": _summary(mad),
        "direct_dominant_mismatch_rate": float(sum(dom) / len(dom)),
        "direct_tail_rate_tv_ge_035": float(sum(v >= 0.35 for v in tv) / len(tv)),
        "acceptance_rate": _summary(rates),
        "proposals_max_per_pair": _summary(props),
    }


def _group_direct(pairs: Sequence[dict], key: str) -> dict:
    groups = defaultdict(list)
    for row in pairs:
        groups[str(row[key])].append(row)
    return {name: _aggregate_direct(rows) for name, rows in sorted(groups.items())}


def _decision(direct: dict, by_behavior: dict, by_region: dict, b15_result: dict) -> dict:
    local_valid = bool(direct.get("valid_count") == direct.get("count") == 128)
    cap_ok = bool(int(direct.get("proposal_cap_hits", 0)) == 0)
    if not local_valid:
        classification = "PHASE2B16_INVALID_STOP_AUDIT" if cap_ok else "EXACT_POSTERIOR_REJECTION_COMPUTE_INFEASIBLE"
        route = "STOP_AND_AUDIT_PHASE2B16" if cap_ok else "CLOSE_REJECTION_ROUTE_CONSIDER_STRUCTURAL_REACH_SUPPORT"
        screen = False
    else:
        snis_tv = float(b15_result["pooled"]["posterior_tv"]["mean"])
        direct_tv = float(direct["direct_tv"]["mean"])
        tv_abs = snis_tv - direct_tv
        tv_rel = tv_abs / snis_tv if snis_tv > 0.0 else -math.inf
        snis_sign = float(b15_result["pooled"]["posterior_sign_disagreement_mean"])
        direct_sign = float(direct["direct_sign_disagreement_mean"])
        sign_abs = snis_sign - direct_sign
        sign_rel = sign_abs / snis_sign if snis_sign > 0.0 else -math.inf
        both_behavior = all(
            float(row["direct_tv"]["mean"]) < float(b15_result["by_behavior_seed"][seed]["posterior_tv"]["mean"])
            for seed, row in by_behavior.items()
        )
        region_guard = all(
            float(row["direct_tv"]["mean"]) <= float(b15_result["by_region"][region]["posterior_tv"]["mean"]) + REGION_MAX_DEGRADE_VS_SNIS
            for region, row in by_region.items()
        )
        gates = {
            "snis_tv_material_improvement": bool(tv_abs >= SNIS_TV_ABS_IMPROVEMENT_MIN or tv_rel >= SNIS_TV_REL_IMPROVEMENT_MIN),
            "direct_tv_ceiling": bool(direct_tv <= DIRECT_TV_MAX),
            "snis_sign_material_improvement": bool(sign_abs >= SNIS_SIGN_ABS_IMPROVEMENT_MIN or sign_rel >= SNIS_SIGN_REL_IMPROVEMENT_MIN),
            "direct_tail_ceiling": bool(float(direct["direct_tail_rate_tv_ge_035"]) <= DIRECT_TAIL_MAX),
            "both_behavior_seeds_improve_vs_snis": bool(both_behavior),
            "continuation_region_non_degradation_vs_snis": bool(region_guard),
            "direct_dominant_mismatch_ceiling": bool(float(direct["direct_dominant_mismatch_rate"]) <= DIRECT_DOMINANT_MISMATCH_MAX),
        }
        screen = bool(all(gates.values()))
        if screen:
            classification = "EXACT_REJECTION_POSTERIOR_CONTINUATION_SUPPORTED"
            route = "PRECOMMIT_SMALL_EXACT_POSTERIOR_CONTINUATION_TARGET_TRAINING_PILOT"
        else:
            classification = "EXACT_POSTERIOR_STILL_TOO_UNSTABLE_CLOSE_ESTIMATOR_REPAIR_PATH"
            route = "CLOSE_ESTIMATOR_REPAIR_CHOOSE_STRUCTURAL_REACH_SUPPORT_OR_CERTIFIED_V1_FALLBACK"
        return {
            "classification": classification, "next_route": route,
            "local_valid": local_valid, "proposal_cap_pass": cap_ok,
            "snis_reference_mean_tv": snis_tv, "direct_mean_tv": direct_tv,
            "snis_to_direct_tv_absolute_improvement": float(tv_abs),
            "snis_to_direct_tv_relative_improvement": float(tv_rel),
            "snis_reference_sign_disagreement": snis_sign, "direct_sign_disagreement": direct_sign,
            "snis_to_direct_sign_absolute_improvement": float(sign_abs),
            "snis_to_direct_sign_relative_improvement": float(sign_rel),
            **gates,
            "screen_pass": screen,
            "small_training_pilot_precommit_allowed": bool(screen),
            "training_authorized": False, "full_x4_confirmation_authorized": False,
            "architecture_winner_selected": False, "production_training_authorized": False,
            "ready_for_tables": False,
        }
    return {
        "classification": classification, "next_route": route,
        "local_valid": local_valid, "proposal_cap_pass": cap_ok,
        "screen_pass": screen, "small_training_pilot_precommit_allowed": False,
        "training_authorized": False, "full_x4_confirmation_authorized": False,
        "architecture_winner_selected": False, "production_training_authorized": False,
        "ready_for_tables": False,
    }


def run(args) -> dict:
    repo_root = Path(args.repo_root).resolve()
    solver_path = Path(args.solver).resolve()
    heldout_root = Path(args.heldout_root).resolve()
    b13_root = Path(args.phase2b13_root).resolve()
    b13_result_path = Path(args.phase2b13_result).resolve()
    b14_result_path = Path(args.phase2b14_result).resolve()
    b15_root = Path(args.phase2b15_root).resolve()
    b15_result_path = Path(args.phase2b15_result).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if _sha256(b15_result_path) != B15_RESULT_SHA256:
        raise RuntimeError("Phase2B16 Phase2B15 result SHA drift")
    j15 = json.loads(b15_result_path.read_text(encoding="utf-8"))
    if j15.get("schema") != b15.SCHEMA or j15.get("status") != B15_STATUS:
        raise RuntimeError("Phase2B16 requires exact failed Phase2B15 result")
    if bool((j15.get("decision") or {}).get("screen_pass")):
        raise RuntimeError("Phase2B16 cannot follow a passing Phase2B15")

    _j13, j14 = b15._validate_source_results(b13_result_path, b14_result_path)
    anchors, heldout_identity = b15._select_anchors(heldout_root, j14)
    if len(anchors) != 64:
        raise RuntimeError("Phase2B16 anchor count drift")
    b15_rows = _load_b15_partials(b15_root, anchors)
    _b15_pairs, b15_reproduced = _verify_b15_baseline(b15_rows, j15)

    all_rows = []
    behavior_identity = []
    for behavior_seed in map(int, TRAINING_SEEDS):
        checkpoint = b13_root / b13.CANDIDATE_ARM / f"seed_{behavior_seed}" / "resume_checkpoint.pt"
        states, identity = b15._load_behavior_states(checkpoint, behavior_seed)
        behavior_identity.append(identity)
        rows = _run_behavior_seed(
            repo_root=repo_root, solver_path=solver_path, output_root=output_root,
            execution_sha=str(args.execution_sha), behavior_seed=behavior_seed,
            behavior_states=states, anchors=anchors, workers=int(args.workers),
        )
        all_rows.extend(rows)

    pairs = _pair_direct(all_rows)
    direct = _aggregate_direct(pairs)
    by_behavior = _group_direct(pairs, "behavior_seed")
    by_region = _group_direct(pairs, "region")
    by_evaluation = _group_direct(pairs, "evaluation_seed")
    decision = _decision(direct, by_behavior, by_region, j15)

    return {
        "schema": SCHEMA,
        "status": decision["classification"],
        "execution_sha": str(args.execution_sha),
        "domain": DOMAIN,
        "representation": b15.REPRESENTATION,
        "contract": {
            "regions": list(REGIONS), "k_accepted_per_block": K, "blocks": BLOCKS,
            "anchors": len(anchors), "behavior_seeds": list(map(int, TRAINING_SEEDS)),
            "evaluation_seeds": list(map(int, EVALUATION_SEEDS)),
            "max_proposals_per_task": MAX_PROPOSALS_PER_TASK,
            "sampler": "EXACT_REJECTION_FROM_UNIFORM_PRIVATE_PRIOR_USING_FROZEN_ACTION_PATH_LIKELIHOOD",
            "future_board_sampled_after_private_acceptance": True,
            "likelihood_floor": None, "weight_clipping": None, "mcmc": False, "sir": False,
        },
        "source_phase2b15_result_sha256": B15_RESULT_SHA256,
        "frozen_inputs": {"heldout": heldout_identity, "behavior_checkpoints": behavior_identity},
        "phase2b15_reproduced": b15_reproduced,
        "direct_posterior": direct,
        "by_behavior_seed": by_behavior,
        "by_region": by_region,
        "by_evaluation_seed": by_evaluation,
        "decision": decision,
        "training_authorized": False, "full_x4_confirmation_authorized": False,
        "architecture_winner_selected": False, "production_training_authorized": False,
        "ready_for_tables": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase2B16 exact rejection posterior continuation screen")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--phase2b13-root", type=Path, required=True)
    parser.add_argument("--phase2b13-result", type=Path, required=True)
    parser.add_argument("--phase2b14-result", type=Path, required=True)
    parser.add_argument("--phase2b15-root", type=Path, required=True)
    parser.add_argument("--phase2b15-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()
    if int(args.workers) < 1 or int(args.workers) > MAX_WORKERS:
        raise RuntimeError("Phase2B16 workers outside frozen range")
    result = run(args)
    out = Path(args.output_root).resolve() / "R7_5_ARCH_RESET_V1PLUS_PHASE2B16_EXACT_REJECTION_POSTERIOR_CONTINUATION.json"
    _atomic_json(result, out)
    direct = result.get("direct_posterior") or {}
    print(json.dumps({
        "status": result["status"],
        "direct_mean_tv": (direct.get("direct_tv") or {}).get("mean"),
        "direct_sign_disagreement": direct.get("direct_sign_disagreement_mean"),
        "direct_tail_rate_tv_ge_035": direct.get("direct_tail_rate_tv_ge_035"),
        "acceptance_rate_median": (direct.get("acceptance_rate") or {}).get("p50"),
        "proposal_cap_hits": direct.get("proposal_cap_hits"),
        "screen_pass": result["decision"]["screen_pass"],
        "next_route": result["decision"]["next_route"],
        "result": str(out), "result_sha256": _sha256(out),
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
