from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from spincore.deep_cfr import _batch, regret_matching_policy
from spincore.r7 import stratified_audit_indices, weighted_mean_tv
from spincore.solver import SolverLibrary
from spincore_nn import AveragePolicyNet, UniformReservoir
from spincore_nn.codec import collate_inputs, decode_spnniv1
from spincore_nn.reservoir import StrategySample
from spincore_nn.training import train_step

from run_r7_3_advantage_fit_sign_sensitivity import (
    DEFAULT_DECK_STREAM_SEED,
    _predict,
    collect_common_memory,
    train_replica,
)


SPECS = tuple((0xB3100 + i * 0x101, 0xC4100 + i * 0x211) for i in range(4))


def _q(xs, q):
    if not xs:
        return math.inf
    t = torch.tensor(xs, dtype=torch.float32)
    return float(torch.quantile(t, torch.tensor(float(q))))


def _regret_target(sample):
    legal = tuple(i for i, yes in enumerate(sample.legal) if yes)
    return tuple(regret_matching_policy(sample.target, legal))


def _make_strategy_memory(adv_memory):
    out = UniformReservoir(max(len(adv_memory.items), 1), 0xBEE5A11)
    out.items = [
        StrategySample(
            sample.observation,
            sample.legal,
            _regret_target(sample),
            float(sample.weight),
            int(sample.iteration),
        )
        for sample in adv_memory.items
    ]
    out.seen = len(out.items)
    return out


def _policy_fit_tv(model, memory, ids, device):
    samples = [memory.items[i] for i in ids]
    batch = collate_inputs([decode_spnniv1(x.observation) for x in samples], device=device)
    target = torch.tensor([x.target for x in samples], dtype=torch.float32, device=device)
    weights = torch.tensor([x.weight for x in samples], dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        pred = model.probabilities(batch)
    return float(weighted_mean_tv(pred, target, weights))


def _train_behavior(memory, *, config, init_seed, batch_seed, args):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(init_seed))
        model = AveragePolicyNet(config).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    rng = random.Random(int(batch_seed))
    audit_ids = stratified_audit_indices(len(memory.items), int(args.audit_size), 0xB3A110 ^ int(init_seed))
    steps = 0
    progress = []
    while steps < int(args.max_steps):
        chunk = min(int(args.chunk_steps), int(args.max_steps) - steps)
        losses = []
        for _ in range(chunk):
            samples = memory.sample(min(int(args.batch_size), len(memory.items)), rng)
            batch, target, weights = _batch(samples, args.device)
            losses.append(float(train_step(model, opt, batch, target, weights, "strategy")))
        steps += chunk
        tv = _policy_fit_tv(model, memory, audit_ids, args.device)
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
        "final_weighted_mean_tv": float(progress[-1]["weighted_mean_tv"]),
        "progress": progress,
    }


def _policy_predictions(model, observations, device):
    batch = collate_inputs([decode_spnniv1(x) for x in observations], device=device)
    model.eval()
    with torch.no_grad():
        return model.probabilities(batch).detach().cpu()


def _hard_adv_predictions(model, observations, legal_masks, device):
    raw = _predict(model, observations, device)
    rows = []
    for i, legal_mask in enumerate(legal_masks):
        legal = tuple(a for a, yes in enumerate(legal_mask) if yes)
        rows.append(regret_matching_policy(raw[i].tolist(), legal))
    return torch.tensor(rows, dtype=torch.float32)


def _pairwise(predictions):
    rows = []
    for left in range(len(predictions)):
        for right in range(left + 1, len(predictions)):
            tv = 0.5 * torch.abs(predictions[left] - predictions[right]).sum(1)
            rows.append({
                "left": int(left),
                "right": int(right),
                "mean_tv": float(tv.mean()),
                "p50_tv": float(torch.quantile(tv, torch.tensor(0.50))),
                "p95_tv": float(torch.quantile(tv, torch.tensor(0.95))),
                "max_tv": float(tv.max()),
            })
    return {
        "pairs": rows,
        "mean_tv_average": sum(x["mean_tv"] for x in rows) / max(len(rows), 1),
        "p95_tv_average": sum(x["p95_tv"] for x in rows) / max(len(rows), 1),
        "max_tv_average": sum(x["max_tv"] for x in rows) / max(len(rows), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare Advantage+hard-RM fits with direct regret-policy surrogate fits on identical memory")
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument("--out", type=Path, default=Path("validation/R7_3_BEHAVIOR_SURROGATE_STABILITY_256.json"))
    ap.add_argument("--roots", type=int, default=256)
    ap.add_argument("--deck-stream-seed", type=int, default=DEFAULT_DECK_STREAM_SEED)
    ap.add_argument("--reservoir-capacity", type=int, default=100000)
    ap.add_argument("--chunk-steps", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=4096)
    ap.add_argument("--fit-target", type=float, default=0.08)
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
    strategy_memory = _make_strategy_memory(adv_memory)
    eval_ids = stratified_audit_indices(len(adv_memory.items), int(args.eval_size), 0xB34A710)
    eval_samples = [adv_memory.items[i] for i in eval_ids]
    observations = [x.observation for x in eval_samples]
    legal_masks = [tuple(int(v) for v in x.legal) for x in eval_samples]

    advantage_predictions = []
    advantage_reports = []
    behavior_predictions = []
    behavior_reports = []
    config = None
    for init_seed, batch_seed in SPECS:
        bundle, adv_report = train_replica(
            memory_state=adv_memory.state_dict(),
            init_seed=int(init_seed),
            batch_seed=int(batch_seed),
            args=args,
        )
        config = bundle.config
        advantage_predictions.append(_hard_adv_predictions(bundle.advantage, observations, legal_masks, args.device))
        advantage_reports.append(adv_report)
        behavior, behavior_report = _train_behavior(
            strategy_memory,
            config=bundle.config,
            init_seed=int(init_seed),
            batch_seed=int(batch_seed),
            args=args,
        )
        behavior_predictions.append(_policy_predictions(behavior, observations, args.device))
        behavior_reports.append(behavior_report)

    advantage_pairwise = _pairwise(advantage_predictions)
    behavior_pairwise = _pairwise(behavior_predictions)
    payload = {
        "schema": "SPINCORE_R7_3_BEHAVIOR_SURROGATE_STABILITY_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "collection": collection,
        "same_advantage_memory": True,
        "regret_policy_targets_derived_from_each_advantage_sample": True,
        "advantage_fit_reports": advantage_reports,
        "behavior_surrogate_fit_reports": behavior_reports,
        "advantage_hard_regret_pairwise": advantage_pairwise,
        "direct_behavior_pairwise": behavior_pairwise,
        "summary": {
            "advantage_hard_regret_mean_tv": float(advantage_pairwise["mean_tv_average"]),
            "advantage_hard_regret_p95_tv": float(advantage_pairwise["p95_tv_average"]),
            "behavior_surrogate_mean_tv": float(behavior_pairwise["mean_tv_average"]),
            "behavior_surrogate_p95_tv": float(behavior_pairwise["p95_tv_average"]),
            "behavior_to_advantage_mean_ratio": float(behavior_pairwise["mean_tv_average"] / max(advantage_pairwise["mean_tv_average"], 1e-12)),
            "behavior_to_advantage_p95_ratio": float(behavior_pairwise["p95_tv_average"] / max(advantage_pairwise["p95_tv_average"], 1e-12)),
            "diagnosis": (
                "DIRECT_BEHAVIOR_SURROGATE_MATERIAL_SAME_MEMORY"
                if min(
                    behavior_pairwise["mean_tv_average"] / max(advantage_pairwise["mean_tv_average"], 1e-12),
                    behavior_pairwise["p95_tv_average"] / max(advantage_pairwise["p95_tv_average"], 1e-12),
                ) <= 0.70
                else "DIRECT_BEHAVIOR_SURROGATE_NOT_MATERIAL_SAME_MEMORY"
            ),
        },
        "interpretation_note": (
            "Diagnostic only. The direct behavior surrogate is trained on hard-regret-matching "
            "policies derived from each stored Advantage target, using the same observations and "
            "LCFR weights. It is not assumed to be theoretically equivalent to Deep CFR because "
            "regret matching is nonlinear and sample-level mapping need not commute with cumulative "
            "regret estimation. It is a screening test for whether predicting behavior directly "
            "removes the sign-support instability seen in Advantage regression."
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
