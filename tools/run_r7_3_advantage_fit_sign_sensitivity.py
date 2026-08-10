from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from spincore.deep_cfr import DeepCFRDomainSession, icm_delta_utility, regret_matching_policy
from spincore.solver import SolverLibrary
from spincore.r7 import stratified_audit_indices
from spincore_nn.codec import collate_inputs, decode_spnniv1
from spincore_nn.reservoir import UniformReservoir

from run_r7_3_diagnostic import hu_episode, make_bundle
from run_r7_3_variance_decomposition import _advantage_fit_nrmse, _finite


PAYOUT = (0.5, 0.3, 0.2)
DEFAULT_COLLECTION_SEED = 20260829
DEFAULT_DECK_STREAM_SEED = 0xA6F17EED
REPLICA_SPECS = (
    (0x11111, 0xAAAA1),
    (0x22222, 0xBBBB2),
    (0x33333, 0xCCCC3),
    (0x44444, 0xDDDD4),
)
EPSILONS = (0.01, 0.05, 0.10)
FRAGILITY_THRESHOLDS = (0.02, 0.05, 0.10, 0.25)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return math.inf
    xs = sorted(float(x) for x in values)
    pos = max(0.0, min(1.0, float(q))) * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    t = pos - lo
    return xs[lo] * (1.0 - t) + xs[hi] * t


def _predict(model, observations: list[bytes], device: str) -> torch.Tensor:
    batch = collate_inputs([decode_spnniv1(x) for x in observations], device=device)
    model.eval()
    with torch.no_grad():
        return model(batch).detach().cpu()


def _epsilon_floor_policy(values: list[float], legal: tuple[int, ...], epsilon: float) -> list[float]:
    scale = math.sqrt(sum(float(values[a]) ** 2 for a in legal) / max(len(legal), 1))
    floor = float(epsilon) * max(scale, 1e-8)
    weights = [0.0] * len(values)
    total = 0.0
    for action in legal:
        w = max(float(values[action]), 0.0) + floor
        weights[action] = w
        total += w
    if total <= 0.0:
        p = 1.0 / len(legal)
        return [p if action in legal else 0.0 for action in range(len(values))]
    return [w / total for w in weights]


def _pair_metrics(pred_a: torch.Tensor, pred_b: torch.Tensor, legal_masks: list[tuple[int, ...]]) -> dict:
    tvs: list[float] = []
    sign_disagreement_rates: list[float] = []
    support_equal: list[float] = []
    fragility: list[float] = []
    epsilon_tvs: dict[float, list[float]] = {eps: [] for eps in EPSILONS}

    for i, legal_mask in enumerate(legal_masks):
        legal = tuple(a for a, yes in enumerate(legal_mask) if yes)
        va = [float(x) for x in pred_a[i].tolist()]
        vb = [float(x) for x in pred_b[i].tolist()]
        pa = regret_matching_policy(va, legal)
        pb = regret_matching_policy(vb, legal)
        tv = 0.5 * sum(abs(x - y) for x, y in zip(pa, pb))
        tvs.append(tv)

        signs_a = tuple(va[a] > 0.0 for a in legal)
        signs_b = tuple(vb[a] > 0.0 for a in legal)
        sign_disagreement_rates.append(
            sum(1.0 for x, y in zip(signs_a, signs_b) if x != y) / max(len(legal), 1)
        )
        support_equal.append(1.0 if signs_a == signs_b else 0.0)

        def normalized_min_abs(values: list[float]) -> float:
            scale = math.sqrt(sum(values[a] ** 2 for a in legal) / max(len(legal), 1))
            return min(abs(values[a]) for a in legal) / max(scale, 1e-8)

        fragility.append(min(normalized_min_abs(va), normalized_min_abs(vb)))
        for eps in EPSILONS:
            ea = _epsilon_floor_policy(va, legal, eps)
            eb = _epsilon_floor_policy(vb, legal, eps)
            epsilon_tvs[eps].append(0.5 * sum(abs(x - y) for x, y in zip(ea, eb)))

    total_tv = sum(tvs)
    conditioned = {}
    for threshold in FRAGILITY_THRESHOLDS:
        ids = [i for i, value in enumerate(fragility) if value <= threshold]
        conditioned[str(threshold)] = {
            "observation_fraction": len(ids) / max(len(tvs), 1),
            "mean_tv": sum(tvs[i] for i in ids) / max(len(ids), 1) if ids else 0.0,
            "fraction_of_total_tv_mass": (
                sum(tvs[i] for i in ids) / max(total_tv, 1e-12) if ids else 0.0
            ),
        }

    return {
        "observation_count": len(tvs),
        "regret_matching_mean_tv": sum(tvs) / max(len(tvs), 1),
        "regret_matching_p50_tv": _quantile(tvs, 0.50),
        "regret_matching_p95_tv": _quantile(tvs, 0.95),
        "regret_matching_max_tv": max(tvs) if tvs else math.inf,
        "mean_legal_sign_disagreement_rate": sum(sign_disagreement_rates) / max(len(tvs), 1),
        "positive_support_equal_fraction": sum(support_equal) / max(len(tvs), 1),
        "normalized_min_abs_advantage_p50": _quantile(fragility, 0.50),
        "normalized_min_abs_advantage_p95": _quantile(fragility, 0.95),
        "fragility_conditioning": conditioned,
        "epsilon_floor_policy_tv": {
            str(eps): {
                "mean_tv": sum(vals) / max(len(vals), 1),
                "p95_tv": _quantile(vals, 0.95),
                "mean_ratio_to_hard_regret_matching": (
                    (sum(vals) / max(len(vals), 1))
                    / max(sum(tvs) / max(len(tvs), 1), 1e-12)
                ),
            }
            for eps, vals in epsilon_tvs.items()
        },
    }


def collect_common_memory(*, solver, roots: int, deck_stream_seed: int, reservoir_capacity: int, device: str):
    bundle = make_bundle(
        DEFAULT_COLLECTION_SEED,
        device=device,
        reservoir_capacity=int(reservoir_capacity),
        lr=1e-3,
    )
    session = DeepCFRDomainSession(
        solver_library=solver,
        bundle=bundle,
        terminal_utility=icm_delta_utility(PAYOUT),
        device=device,
    )
    if session.behavior.ready:
        raise RuntimeError("bootstrap Advantage policy unexpectedly ready")
    session.collector.rng = random.Random(0xC011EC7)
    episode = hu_episode()
    live = [i for i, stack in enumerate(episode.stacks) if stack > 0]
    nodes = 0
    for root_index in range(int(roots)):
        deck_seed = (
            int(deck_stream_seed) * 1_000_003 + root_index * 97 + 1
        ) & ((1 << 64) - 1)
        for traverser in live:
            root = solver.create(episode, int(deck_seed))
            try:
                result = session.collector.collect_advantage(
                    root,
                    traverser=int(traverser),
                    iteration=1,
                )
            finally:
                root.close()
            nodes += int(result.nodes)
    return bundle.adv_mem, {
        "roots": int(roots),
        "advantage_samples": len(bundle.adv_mem.items),
        "advantage_seen": int(bundle.adv_mem.seen),
        "nodes": int(nodes),
        "behavior": "EXACT_ZERO_REGRET_UNIFORM",
    }


def train_replica(*, memory_state: dict, init_seed: int, batch_seed: int, args):
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
    audit_seed = 0xA0D17
    while local_steps < int(args.max_steps):
        steps = min(int(args.chunk_steps), int(args.max_steps) - local_steps)
        session.train_advantage(steps=steps, batch_size=int(args.batch_size))
        local_steps += steps
        nrmse = _advantage_fit_nrmse(
            bundle,
            sample_size=int(args.audit_size),
            seed=audit_seed,
            device=args.device,
        )
        progress.append({"optimizer_steps": local_steps, "weighted_nrmse": float(nrmse)})
        if _finite(nrmse) and float(nrmse) <= float(args.fit_target):
            break
    return bundle, {
        "init_seed": int(init_seed),
        "batch_seed": int(batch_seed),
        "optimizer_steps": int(local_steps),
        "final_weighted_nrmse": float(progress[-1]["weighted_nrmse"]),
        "progress": progress,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Same-memory Advantage fit/regret-sign sensitivity")
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument("--out", type=Path, default=Path("validation/R7_3_ADVANTAGE_FIT_SIGN_SENSITIVITY_256.json"))
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
    memory_state = memory.state_dict()
    ids = stratified_audit_indices(len(memory.items), int(args.eval_size), 0xE7A1)
    samples = [memory.items[i] for i in ids]
    observations = [sample.observation for sample in samples]
    legal_masks = [tuple(int(x) for x in sample.legal) for sample in samples]

    bundles = []
    replica_reports = []
    predictions = []
    for init_seed, batch_seed in REPLICA_SPECS:
        bundle, report = train_replica(
            memory_state=memory_state,
            init_seed=int(init_seed),
            batch_seed=int(batch_seed),
            args=args,
        )
        bundles.append(bundle)
        replica_reports.append(report)
        predictions.append(_predict(bundle.advantage, observations, args.device))

    pair_reports = {}
    pair_means = []
    pair_p95s = []
    support_equal = []
    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            key = f"replica_{i}_vs_{j}"
            metrics = _pair_metrics(predictions[i], predictions[j], legal_masks)
            pair_reports[key] = metrics
            pair_means.append(float(metrics["regret_matching_mean_tv"]))
            pair_p95s.append(float(metrics["regret_matching_p95_tv"]))
            support_equal.append(float(metrics["positive_support_equal_fraction"]))

    mean_pair_tv = sum(pair_means) / max(len(pair_means), 1)
    mean_support_equal = sum(support_equal) / max(len(support_equal), 1)
    payload = {
        "schema": "SPINCORE_R7_3_ADVANTAGE_FIT_SIGN_SENSITIVITY_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "collection": collection,
        "same_memory_for_all_replicas": True,
        "eval_observation_count": len(observations),
        "replicas": replica_reports,
        "pairwise": pair_reports,
        "summary": {
            "pairwise_regret_matching_mean_tv_average": float(mean_pair_tv),
            "pairwise_regret_matching_p95_tv_average": sum(pair_p95s) / max(len(pair_p95s), 1),
            "positive_support_equal_fraction_average": float(mean_support_equal),
            "diagnosis": (
                "ADVANTAGE_FIT_REGRET_SIGN_INSTABILITY_MATERIAL"
                if mean_pair_tv >= 0.15 or mean_support_equal <= 0.75
                else "ADVANTAGE_FIT_REGRET_SIGN_INSTABILITY_NOT_MATERIAL_AT_SCREEN_SCALE"
            ),
        },
        "interpretation_note": (
            "All AdvantageNet replicas train on the exact same frozen external-sampling memory. "
            "Only neural initialization and minibatch order differ. Pairwise model outputs are "
            "converted through the production hard regret-matching rule, then compared. The "
            "fragility audit measures how much TV is concentrated where legal predicted regrets "
            "lie near zero; epsilon-floor mappings are post-hoc sensitivity probes only and do "
            "not change production CFR semantics."
        ),
        "acceptance_gate_changed": False,
        "production_regret_matching_changed": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
