from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

from spincore.deep_cfr import _batch, regret_matching_policy
from spincore.r7 import stratified_audit_indices, weighted_mean_tv
from spincore.solver import SolverLibrary
from spincore_nn import AveragePolicyNet, UniformReservoir
from spincore_nn.codec import collate_inputs, decode_spnniv1
from spincore_nn.reservoir import StrategySample
from spincore_nn.training import train_step

from run_r7_3_advantage_fit_sign_sensitivity import DEFAULT_DECK_STREAM_SEED, collect_common_memory


REPLICAS = tuple((0xA7100 + i * 0x101, 0xE3100 + i * 0x211) for i in range(4))


def _sample_rm_memory(adv_memory):
    out = UniformReservoir(max(len(adv_memory.items), 1), 0xA771)
    out.items = []
    for sample in adv_memory.items:
        legal = tuple(i for i, yes in enumerate(sample.legal) if yes)
        out.items.append(
            StrategySample(
                sample.observation,
                sample.legal,
                tuple(regret_matching_policy(sample.target, legal)),
                float(sample.weight),
                int(sample.iteration),
            )
        )
    out.seen = len(out.items)
    return out


def _aggregated_memories(adv_memory):
    groups = {}
    for sample in adv_memory.items:
        key = (sample.observation, tuple(int(x) for x in sample.legal))
        if key not in groups:
            groups[key] = {
                "weight": 0.0,
                "regret": [0.0] * 6,
                "policy": [0.0] * 6,
                "iteration": int(sample.iteration),
                "count": 0,
            }
        row = groups[key]
        w = float(sample.weight)
        row["weight"] += w
        row["count"] += 1
        row["iteration"] = max(int(row["iteration"]), int(sample.iteration))
        legal = tuple(i for i, yes in enumerate(sample.legal) if yes)
        rm = regret_matching_policy(sample.target, legal)
        for action in range(6):
            row["regret"][action] += w * float(sample.target[action])
            row["policy"][action] += w * float(rm[action])

    regret_then_rm = UniformReservoir(max(len(groups), 1), 0xA772)
    mean_rm = UniformReservoir(max(len(groups), 1), 0xA773)
    regret_then_rm.items = []
    mean_rm.items = []
    duplicate_counts = []
    for (observation, legal_mask), row in groups.items():
        total = float(row["weight"])
        if total <= 0.0:
            continue
        mean_regret = [x / total for x in row["regret"]]
        mean_policy = [x / total for x in row["policy"]]
        legal = tuple(i for i, yes in enumerate(legal_mask) if yes)
        regret_policy = tuple(regret_matching_policy(mean_regret, legal))
        regret_then_rm.items.append(
            StrategySample(
                observation,
                legal_mask,
                regret_policy,
                total,
                int(row["iteration"]),
            )
        )
        mean_rm.items.append(
            StrategySample(
                observation,
                legal_mask,
                tuple(mean_policy),
                total,
                int(row["iteration"]),
            )
        )
        duplicate_counts.append(int(row["count"]))
    regret_then_rm.seen = len(regret_then_rm.items)
    mean_rm.seen = len(mean_rm.items)
    stats = {
        "raw_samples": len(adv_memory.items),
        "unique_observation_legal_groups": len(groups),
        "compression_ratio": len(groups) / max(len(adv_memory.items), 1),
        "groups_with_duplicates": sum(1 for x in duplicate_counts if x > 1),
        "max_group_count": max(duplicate_counts) if duplicate_counts else 0,
        "mean_group_count": sum(duplicate_counts) / max(len(duplicate_counts), 1),
    }
    return regret_then_rm, mean_rm, stats


def _fit_tv(model, memory, *, sample_size, seed, device):
    ids = stratified_audit_indices(len(memory.items), int(sample_size), int(seed))
    if not ids:
        return math.inf
    samples = [memory.items[i] for i in ids]
    batch = collate_inputs([decode_spnniv1(x.observation) for x in samples], device=device)
    target = torch.tensor([x.target for x in samples], dtype=torch.float32, device=device)
    weights = torch.tensor([x.weight for x in samples], dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        pred = model.probabilities(batch)
    return float(weighted_mean_tv(pred, target, weights))


def _train(memory, *, config, init_seed, batch_seed, args, audit_seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(init_seed))
        model = AveragePolicyNet(config).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    rng = random.Random(int(batch_seed))
    steps = 0
    progress = []
    while steps < int(args.max_steps):
        chunk = min(int(args.chunk_steps), int(args.max_steps) - steps)
        losses = []
        for _ in range(chunk):
            samples = memory.sample(min(int(args.batch_size), len(memory.items)), rng)
            batch, target, weights = _batch(samples, args.device)
            losses.append(float(train_step(model, optimizer, batch, target, weights, "strategy")))
        steps += chunk
        tv = _fit_tv(model, memory, sample_size=int(args.audit_size), seed=audit_seed, device=args.device)
        progress.append({
            "optimizer_steps": int(steps),
            "weighted_mean_tv": float(tv),
            "mean_training_loss": sum(losses) / max(len(losses), 1),
            "fit_target_reached": bool(math.isfinite(tv) and tv <= float(args.fit_target)),
        })
        if progress[-1]["fit_target_reached"]:
            break
    return model, {
        "init_seed": int(init_seed),
        "batch_seed": int(batch_seed),
        "optimizer_steps": int(steps),
        "final_own_target_weighted_mean_tv": float(progress[-1]["weighted_mean_tv"]),
        "progress": progress,
    }


def _predictions(model, observations, device):
    batch = collate_inputs([decode_spnniv1(x) for x in observations], device=device)
    model.eval()
    with torch.no_grad():
        return model.probabilities(batch).detach().cpu()


def _pairwise(predictions):
    rows = []
    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            tv = 0.5 * torch.abs(predictions[i] - predictions[j]).sum(1)
            rows.append({
                "left": i,
                "right": j,
                "mean_tv": float(tv.mean()),
                "p95_tv": float(torch.quantile(tv, torch.tensor(0.95))),
                "max_tv": float(tv.max()),
            })
    return {
        "pairs": rows,
        "mean_tv_average": sum(x["mean_tv"] for x in rows) / max(len(rows), 1),
        "p95_tv_average": sum(x["p95_tv"] for x in rows) / max(len(rows), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare sample-RM, mean-RM and RM-of-mean-regret behavior targets")
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument("--out", type=Path, default=Path("validation/R7_3_BEHAVIOR_TARGET_AGGREGATION_256.json"))
    ap.add_argument("--roots", type=int, default=256)
    ap.add_argument("--deck-stream-seed", type=int, default=DEFAULT_DECK_STREAM_SEED)
    ap.add_argument("--reservoir-capacity", type=int, default=100000)
    ap.add_argument("--chunk-steps", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=4096)
    ap.add_argument("--fit-target", type=float, default=0.105)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--audit-size", type=int, default=1024)
    ap.add_argument("--eval-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    solver = SolverLibrary(args.solver)
    args._solver_obj = solver
    started = time.time()
    adv_memory, collection = collect_common_memory(
        solver=solver,
        roots=int(args.roots),
        deck_stream_seed=int(args.deck_stream_seed),
        reservoir_capacity=int(args.reservoir_capacity),
        device=args.device,
    )
    sample_rm = _sample_rm_memory(adv_memory)
    regret_mean_rm, mean_rm, aggregation = _aggregated_memories(adv_memory)
    memories = {
        "sample_level_rm": sample_rm,
        "mean_of_sample_rm": mean_rm,
        "rm_of_weighted_mean_regret": regret_mean_rm,
    }

    eval_ids = stratified_audit_indices(len(sample_rm.items), int(args.eval_size), 0xA6601)
    observations = [sample_rm.items[i].observation for i in eval_ids]
    results = {}
    config = None
    for mode_index, (mode, memory) in enumerate(memories.items()):
        predictions = []
        reports = []
        for replica, (init_seed, batch_seed) in enumerate(REPLICAS):
            if config is None:
                # create a tiny bundle only to recover the canonical NetworkConfig
                from run_r7_3_diagnostic import make_bundle
                tmp = make_bundle(1, device=args.device, reservoir_capacity=1, lr=float(args.lr))
                config = tmp.config
            model, report = _train(
                memory,
                config=config,
                init_seed=int(init_seed),
                batch_seed=int(batch_seed),
                args=args,
                audit_seed=0xA6602 ^ (mode_index * 0x101) ^ replica,
            )
            report["fit_on_raw_sample_rm_tv"] = _fit_tv(
                model,
                sample_rm,
                sample_size=int(args.audit_size),
                seed=0xA6603 ^ replica,
                device=args.device,
            )
            predictions.append(_predictions(model, observations, args.device))
            reports.append(report)
        results[mode] = {
            "memory_items": len(memory.items),
            "replicas": reports,
            "pairwise": _pairwise(predictions),
            "mean_raw_sample_rm_fit_tv": sum(r["fit_on_raw_sample_rm_tv"] for r in reports) / len(reports),
        }

    baseline = results["sample_level_rm"]["pairwise"]
    candidate = results["rm_of_weighted_mean_regret"]["pairwise"]
    payload = {
        "schema": "SPINCORE_R7_3_BEHAVIOR_TARGET_AGGREGATION_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "collection": collection,
        "aggregation": aggregation,
        "results": results,
        "summary": {
            "sample_level_rm_mean_tv": float(baseline["mean_tv_average"]),
            "sample_level_rm_p95_tv": float(baseline["p95_tv_average"]),
            "rm_of_mean_regret_mean_tv": float(candidate["mean_tv_average"]),
            "rm_of_mean_regret_p95_tv": float(candidate["p95_tv_average"]),
            "rm_of_mean_regret_to_sample_mean_ratio": float(candidate["mean_tv_average"] / max(baseline["mean_tv_average"], 1e-12)),
            "rm_of_mean_regret_to_sample_p95_ratio": float(candidate["p95_tv_average"] / max(baseline["p95_tv_average"], 1e-12)),
        },
        "interpretation_note": (
            "Diagnostic only. mean_of_sample_rm is an exact compression of the weighted cross-entropy "
            "target for duplicate exact observations because cross-entropy is linear in the target. "
            "rm_of_weighted_mean_regret instead first computes the empirical weighted conditional mean "
            "regret vector for each exact observation and then applies hard regret matching; this more "
            "closely mirrors the order implied by weighted-MSE regret regression followed by behavior "
            "mapping, while bypassing unstable Advantage function approximation on repeated states. "
            "Neither direct-policy surrogate is declared equivalent on unseen states."
        ),
        "acceptance_gate_changed": False,
        "production_algorithm_changed": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
