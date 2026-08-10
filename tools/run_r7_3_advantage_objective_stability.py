from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from spincore.deep_cfr import DeepCFRDomainSession, _batch, icm_delta_utility
from spincore.r7 import stratified_audit_indices
from spincore.solver import SolverLibrary
from spincore_nn.reservoir import UniformReservoir

from run_r7_3_advantage_fit_sign_sensitivity import (
    DEFAULT_DECK_STREAM_SEED,
    PAYOUT,
    _pair_metrics,
    _predict,
    collect_common_memory,
)
from run_r7_3_diagnostic import make_bundle
from run_r7_3_variance_decomposition import _advantage_fit_nrmse, _finite


REPLICA_SPECS = (
    (0x51001, 0xA1001),
    (0x52002, 0xA2002),
    (0x53003, 0xA3003),
    (0x54004, 0xA4004),
)
OBJECTIVES = ("mse", "mse_sign", "mse_policy", "mse_sign_policy")


def _per_sample_components(out, target, legal):
    mask = legal.float()
    count = mask.sum(1).clamp_min(1.0)
    mse = (((out - target) ** 2) * mask).sum(1) / count
    target_scale = torch.sqrt(((target**2) * mask).sum(1) / count).clamp_min(1e-4)

    sign_target = (target > 0.0).float()
    sign_logits = out / target_scale[:, None]
    confidence = 0.25 + torch.clamp(torch.abs(target) / target_scale[:, None], 0.0, 1.0)
    sign_per_action = F.binary_cross_entropy_with_logits(sign_logits, sign_target, reduction="none")
    sign_loss = (sign_per_action * confidence * mask).sum(1) / (confidence * mask).sum(1).clamp_min(1e-12)

    positive_target = torch.relu(target) * mask
    pos_sum = positive_target.sum(1, keepdim=True)
    uniform = mask / count[:, None]
    target_policy = torch.where(
        pos_sum > 0.0,
        positive_target / pos_sum.clamp_min(1e-12),
        uniform,
    )
    # Smooth surrogate only for the auxiliary training term. Production behavior
    # remains the unchanged hard ReLU regret-matching rule.
    smooth_positive = F.softplus(out / (0.25 * target_scale[:, None])) * mask
    pred_policy = smooth_positive / smooth_positive.sum(1, keepdim=True).clamp_min(1e-12)
    policy_ce = -(target_policy * torch.log(pred_policy.clamp_min(1e-12))).sum(1)
    return mse, sign_loss, policy_ce


def _train_custom_step(model, optimizer, batch, target, weights, objective: str, aux_weight: float):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    out = model(batch)
    w = weights / weights.mean().clamp_min(1e-12)
    mse, sign_loss, policy_ce = _per_sample_components(out, target, batch["legal"])
    per = mse
    if objective in ("mse_sign", "mse_sign_policy"):
        per = per + float(aux_weight) * sign_loss
    if objective in ("mse_policy", "mse_sign_policy"):
        per = per + float(aux_weight) * policy_ce
    loss = (per * w).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    return float(loss.detach().cpu())


def train_replica(*, memory_state, init_seed: int, batch_seed: int, objective: str, args):
    bundle = make_bundle(
        int(init_seed),
        device=args.device,
        reservoir_capacity=int(args.reservoir_capacity),
        lr=float(args.lr),
    )
    bundle.adv_mem = UniformReservoir.from_state_dict(memory_state)
    bundle.batch_rng = random.Random(int(batch_seed))
    session = DeepCFRDomainSession(
        solver_library=args._solver_obj,
        bundle=bundle,
        terminal_utility=icm_delta_utility(PAYOUT),
        device=args.device,
    )
    session.reset_advantage_network(init_seed=int(init_seed), lr=float(args.lr))
    local_steps = 0
    progress = []
    audit_seed = 0xA0B1EC7
    while local_steps < int(args.max_steps):
        chunk = min(int(args.chunk_steps), int(args.max_steps) - local_steps)
        losses = []
        for _ in range(chunk):
            samples = bundle.adv_mem.sample(
                min(int(args.batch_size), len(bundle.adv_mem.items)), bundle.batch_rng
            )
            batch, target, weights = _batch(samples, args.device)
            losses.append(
                _train_custom_step(
                    bundle.advantage,
                    bundle.adv_opt,
                    batch,
                    target,
                    weights,
                    objective,
                    float(args.aux_weight),
                )
            )
        local_steps += chunk
        nrmse = _advantage_fit_nrmse(
            bundle,
            sample_size=int(args.audit_size),
            seed=audit_seed,
            device=args.device,
        )
        progress.append(
            {
                "optimizer_steps": int(local_steps),
                "weighted_nrmse": float(nrmse),
                "mean_training_loss": sum(losses) / max(len(losses), 1),
            }
        )
        if _finite(nrmse) and float(nrmse) <= float(args.fit_target):
            break
    return bundle, {
        "init_seed": int(init_seed),
        "batch_seed": int(batch_seed),
        "optimizer_steps": int(local_steps),
        "final_weighted_nrmse": float(progress[-1]["weighted_nrmse"]),
        "progress": progress,
    }


def _mean(rows, key):
    return sum(float(row[key]) for row in rows) / max(len(rows), 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Same-memory behavior-aware Advantage objective stability screen")
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--objective", choices=OBJECTIVES, required=True)
    ap.add_argument("--aux-weight", type=float, default=0.20)
    ap.add_argument("--roots", type=int, default=256)
    ap.add_argument("--deck-stream-seed", type=int, default=DEFAULT_DECK_STREAM_SEED)
    ap.add_argument("--reservoir-capacity", type=int, default=100000)
    ap.add_argument("--chunk-steps", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=4096)
    ap.add_argument("--fit-target", type=float, default=0.50)
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
    memory, collection = collect_common_memory(
        solver=solver,
        roots=int(args.roots),
        deck_stream_seed=int(args.deck_stream_seed),
        reservoir_capacity=int(args.reservoir_capacity),
        device=args.device,
    )
    state = memory.state_dict()
    ids = stratified_audit_indices(len(memory.items), int(args.eval_size), 0x0B1EC71)
    samples = [memory.items[i] for i in ids]
    observations = [x.observation for x in samples]
    legal_masks = [tuple(int(v) for v in x.legal) for x in samples]

    predictions = []
    replicas = []
    for init_seed, batch_seed in REPLICA_SPECS:
        bundle, report = train_replica(
            memory_state=state,
            init_seed=int(init_seed),
            batch_seed=int(batch_seed),
            objective=args.objective,
            args=args,
        )
        predictions.append(_predict(bundle.advantage, observations, args.device))
        replicas.append(report)

    pairs = []
    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            row = _pair_metrics(predictions[i], predictions[j], legal_masks)
            row["left"] = i
            row["right"] = j
            pairs.append(row)

    mean_tv = _mean(pairs, "regret_matching_mean_tv")
    mean_p95 = _mean(pairs, "regret_matching_p95_tv")
    support_equal = _mean(pairs, "positive_support_equal_fraction")
    max_nrmse = max(float(r["final_weighted_nrmse"]) for r in replicas)
    payload = {
        "schema": "SPINCORE_R7_3_ADVANTAGE_OBJECTIVE_STABILITY_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "objective": args.objective,
        "aux_weight": float(args.aux_weight),
        "collection": collection,
        "same_memory_for_all_replicas": True,
        "replicas": replicas,
        "pairwise": pairs,
        "summary": {
            "pairwise_regret_matching_mean_tv_average": float(mean_tv),
            "pairwise_regret_matching_p95_tv_average": float(mean_p95),
            "positive_support_equal_fraction_average": float(support_equal),
            "max_replica_weighted_nrmse": float(max_nrmse),
            "frozen_advantage_fit_gate_all_pass": bool(max_nrmse <= 0.75),
        },
        "interpretation_note": (
            "Diagnostic only. Every objective trains four independent AdvantageNets on the same "
            "frozen reservoir. Baseline MSE matches the recovered regression objective. Auxiliary "
            "sign and/or regret-policy terms are smooth training surrogates; evaluation still uses "
            "the unchanged production hard regret-matching map. An objective is interesting only "
            "if it reduces cross-fit regret-policy variance while preserving the frozen Advantage "
            "NRMSE gate."
        ),
        "acceptance_gate_changed": False,
        "production_advantage_objective_changed": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
