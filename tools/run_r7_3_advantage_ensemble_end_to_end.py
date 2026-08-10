from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from spincore.deep_cfr import DeepCFRDomainSession, icm_delta_utility, regret_matching_policy, uniform_policy
from spincore.r7 import FROZEN_GATES, audit_model_fit, cross_seed_policy_tv, stratified_audit_indices
from spincore.solver import SolverLibrary
from spincore_nn.codec import collate_inputs, decode_spnniv1
from spincore_nn.reservoir import UniformReservoir

from run_r7_3_diagnostic import hu_episode, make_bundle, shared_cross_seed_observations
from run_r7_3_path_replication_screen import ADV_RNG_XOR, STRATEGY_RNG_XOR, PAYOUT, _fit_average_policy
from run_r7_3_variance_decomposition import _finite


DEFAULT_SEEDS = (20260829, 20260807)
DEFAULT_SHARED_DECK_STREAM_SEED = 0xD3C5EED
MEMBER_INIT_XOR = 0xE115EED
MEMBER_BATCH_XOR = 0xBA7C8A11


class EnsembleAdvantagePolicy:
    def __init__(self, *, device: str):
        self.models: list[torch.nn.Module] = []
        self.device = device

    @property
    def ready(self) -> bool:
        return bool(self.models)

    def __call__(self, state, observation: bytes, legal: tuple[int, ...]):
        if not self.models:
            return uniform_policy(state, observation, legal)
        batch = collate_inputs([decode_spnniv1(observation)], device=self.device)
        raw = None
        with torch.no_grad():
            for model in self.models:
                model.eval()
                current = model(batch)[0]
                raw = current if raw is None else raw + current
        raw = (raw / float(len(self.models))).detach().cpu().tolist()
        return regret_matching_policy(raw, legal)


def _ensemble_nrmse(models, memory, *, sample_size: int, seed: int, device: str) -> float:
    ids = stratified_audit_indices(len(memory.items), int(sample_size), int(seed))
    if not ids:
        return math.inf
    samples = [memory.items[i] for i in ids]
    batch = collate_inputs([decode_spnniv1(x.observation) for x in samples], device=device)
    target = torch.tensor([x.target for x in samples], dtype=torch.float32, device=device)
    legal = torch.tensor([x.legal for x in samples], dtype=torch.bool, device=device)
    weights = torch.tensor([x.weight for x in samples], dtype=torch.float32, device=device)
    pred = None
    with torch.no_grad():
        for model in models:
            model.eval()
            current = model(batch)
            pred = current if pred is None else pred + current
        pred = pred / float(len(models))
    mask = legal.float()
    count = mask.sum(1).clamp_min(1.0)
    per_sq = (((pred - target) ** 2) * mask).sum(1) / count
    per_energy = ((target**2) * mask).sum(1) / count
    w = weights / weights.mean().clamp_min(1e-12)
    mse = (per_sq * w).sum() / w.sum().clamp_min(1e-12)
    energy = (per_energy * w).sum() / w.sum().clamp_min(1e-12)
    return float(torch.sqrt(mse / energy.clamp_min(1e-12)).cpu())


def _train_member(*, memory_state, algorithm_seed: int, iteration: int, member: int, solver, args):
    init_seed = (
        int(algorithm_seed)
        ^ MEMBER_INIT_XOR
        ^ (int(iteration) * 0x9E3779B1)
        ^ (int(member) * 0x45D9F3B)
    ) & 0x7FFFFFFF
    batch_seed = (
        int(algorithm_seed)
        ^ MEMBER_BATCH_XOR
        ^ (int(iteration) * 0x85EBCA77)
        ^ (int(member) * 0xC2B2AE3D)
    ) & ((1 << 64) - 1)
    bundle = make_bundle(
        int(init_seed),
        device=args.device,
        reservoir_capacity=int(args.reservoir_capacity),
        lr=float(args.lr),
    )
    bundle.adv_mem = UniformReservoir.from_state_dict(memory_state)
    bundle.batch_rng = random.Random(int(batch_seed))
    session = DeepCFRDomainSession(
        solver_library=solver,
        bundle=bundle,
        terminal_utility=icm_delta_utility(PAYOUT),
        device=args.device,
    )
    session.reset_advantage_network(init_seed=int(init_seed), lr=float(args.lr))
    local_steps = 0
    progress = []
    audit_seed = int(algorithm_seed) ^ (iteration * 0x13579B) ^ (member * 0x2468AC)
    from run_r7_3_variance_decomposition import _advantage_fit_nrmse
    while local_steps < int(args.advantage_max_steps_per_iteration):
        steps = min(
            int(args.advantage_chunk_steps),
            int(args.advantage_max_steps_per_iteration) - local_steps,
        )
        session.train_advantage(steps=steps, batch_size=int(args.batch_size))
        local_steps += steps
        nrmse = _advantage_fit_nrmse(
            bundle,
            sample_size=int(args.audit_size),
            seed=audit_seed,
            device=args.device,
        )
        progress.append({"optimizer_steps": int(local_steps), "weighted_nrmse": float(nrmse)})
        if _finite(nrmse) and float(nrmse) <= float(args.advantage_fit_target):
            break
    return bundle.advantage, {
        "member": int(member),
        "init_seed": int(init_seed),
        "batch_seed": int(batch_seed),
        "optimizer_steps": int(local_steps),
        "final_weighted_nrmse": float(progress[-1]["weighted_nrmse"]),
        "progress": progress,
    }


def run_seed(*, seed: int, ensemble_size: int, solver, args):
    bundle = make_bundle(
        int(seed),
        device=args.device,
        reservoir_capacity=int(args.reservoir_capacity),
        lr=float(args.lr),
    )
    session = DeepCFRDomainSession(
        solver_library=solver,
        bundle=bundle,
        terminal_utility=icm_delta_utility(PAYOUT),
        device=args.device,
    )
    behavior = EnsembleAdvantagePolicy(device=args.device)
    session.behavior = behavior
    session.collector.policy = behavior
    advantage_rng = random.Random(int(seed) ^ ADV_RNG_XOR)
    strategy_rng = random.Random(int(seed) ^ STRATEGY_RNG_XOR)
    episode = hu_episode()
    live = [i for i, stack in enumerate(episode.stacks) if stack > 0]
    checkpoints = []
    global_root = 0

    for iteration in range(1, int(args.iterations) + 1):
        for _ in range(int(args.roots_per_iteration)):
            ds = (
                int(args.deck_stream_seed) * 1_000_003 + global_root * 97 + iteration
            ) & ((1 << 64) - 1)
            nodes = advantage_added = strategy_added = 0
            session.collector.rng = advantage_rng
            for traverser in live:
                root = solver.create(episode, int(ds))
                try:
                    result = session.collector.collect_advantage(
                        root, traverser=int(traverser), iteration=int(iteration)
                    )
                finally:
                    root.close()
                nodes += int(result.nodes)
                advantage_added += int(result.samples_added)
            session.collector.rng = strategy_rng
            for target_player in live:
                root = solver.create(episode, int(ds))
                try:
                    strategy_added += int(
                        session.collector.collect_strategy_own_reach(
                            root, target_player=int(target_player), iteration=int(iteration)
                        )
                    )
                finally:
                    root.close()
            c = bundle.counters
            c["iteration"] = max(int(c["iteration"]), int(iteration))
            c["roots"] += 1
            c["nodes"] += int(nodes)
            c["advantage_samples"] += int(advantage_added)
            c["strategy_samples"] += int(strategy_added)
            global_root += 1

        state = bundle.adv_mem.state_dict()
        models = []
        member_reports = []
        for member in range(int(ensemble_size)):
            model, report = _train_member(
                memory_state=state,
                algorithm_seed=int(seed),
                iteration=int(iteration),
                member=int(member),
                solver=solver,
                args=args,
            )
            models.append(model)
            member_reports.append(report)
        behavior.models = models
        ensemble_nrmse = _ensemble_nrmse(
            models,
            bundle.adv_mem,
            sample_size=int(args.audit_size),
            seed=int(seed) ^ (iteration * 0x5EEDBEEF),
            device=args.device,
        )
        checkpoints.append(
            {
                "iteration": int(iteration),
                "roots": int(bundle.counters["roots"]),
                "nodes": int(bundle.counters["nodes"]),
                "advantage_samples": len(bundle.adv_mem.items),
                "advantage_seen": int(bundle.adv_mem.seen),
                "strategy_samples": len(bundle.pol_mem.items),
                "strategy_seen": int(bundle.pol_mem.seen),
                "ensemble_weighted_nrmse": float(ensemble_nrmse),
                "ensemble_frozen_fit_gate_pass": bool(ensemble_nrmse <= FROZEN_GATES["advantage_weighted_nrmse_max"]),
                "members": member_reports,
            }
        )
        print(json.dumps({"seed": seed, "checkpoint": checkpoints[-1]}, sort_keys=True), flush=True)

    # Final AveragePolicy remains a single network; this experiment changes only
    # the Advantage behavior estimator used during CFR collection.
    bundle.batch_rng = random.Random(int(seed) ^ 0xA9E12C7)
    policy_progress, audit = _fit_average_policy(
        bundle=bundle,
        session=session,
        seed=int(seed),
        device=args.device,
        policy_chunk_steps=int(args.policy_chunk_steps),
        policy_max_steps=int(args.policy_max_steps),
        policy_fit_target=float(args.policy_fit_target),
        batch_size=int(args.batch_size),
        audit_size=int(args.audit_size),
    )
    final_policy_audit = audit_model_fit(
        bundle,
        sample_size=max(int(args.audit_size), 2048),
        seed=int(seed) ^ 0x2468ACE0,
        device=args.device,
    )
    final_ensemble_nrmse = _ensemble_nrmse(
        behavior.models,
        bundle.adv_mem,
        sample_size=max(int(args.audit_size), 2048),
        seed=int(seed) ^ 0x13572468,
        device=args.device,
    )
    return bundle, {
        "algorithm_seed": int(seed),
        "ensemble_size": int(ensemble_size),
        "roots": int(bundle.counters["roots"]),
        "nodes": int(bundle.counters["nodes"]),
        "advantage_samples": len(bundle.adv_mem.items),
        "advantage_seen": int(bundle.adv_mem.seen),
        "strategy_samples": len(bundle.pol_mem.items),
        "strategy_seen": int(bundle.pol_mem.seen),
        "checkpoints": checkpoints,
        "policy_progress": policy_progress,
        "final_fit": {
            "ensemble_advantage_weighted_nrmse": float(final_ensemble_nrmse),
            "policy_weighted_mean_tv": float(final_policy_audit["policy_weighted_mean_tv"]),
            "advantage_gate_pass": bool(final_ensemble_nrmse <= FROZEN_GATES["advantage_weighted_nrmse_max"]),
            "policy_gate_pass": bool(float(final_policy_audit["policy_weighted_mean_tv"]) <= FROZEN_GATES["policy_weighted_mean_tv_max"]),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="End-to-end Deep CFR Advantage ensemble screen")
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ensemble-size", type=int, required=True, choices=(1, 2, 4))
    ap.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    ap.add_argument("--deck-stream-seed", type=int, default=DEFAULT_SHARED_DECK_STREAM_SEED)
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--roots-per-iteration", type=int, default=128)
    ap.add_argument("--advantage-chunk-steps", type=int, default=256)
    ap.add_argument("--advantage-max-steps-per-iteration", type=int, default=4096)
    ap.add_argument("--advantage-fit-target", type=float, default=0.50)
    ap.add_argument("--policy-chunk-steps", type=int, default=256)
    ap.add_argument("--policy-max-steps", type=int, default=16384)
    ap.add_argument("--policy-fit-target", type=float, default=0.105)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--audit-size", type=int, default=512)
    ap.add_argument("--cross-seed-per-seed", type=int, default=1024)
    ap.add_argument("--reservoir-capacity", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    if len(seeds) != 2:
        raise SystemExit("requires exactly two algorithm seeds")
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    solver = SolverLibrary(args.solver)
    started = time.time()
    bundles = []
    reports = []
    for seed in seeds:
        bundle, report = run_seed(
            seed=int(seed), ensemble_size=int(args.ensemble_size), solver=solver, args=args
        )
        bundles.append(bundle)
        reports.append(report)
    observations = shared_cross_seed_observations(
        bundles, per_seed=int(args.cross_seed_per_seed), seed=0x715EED
    )
    cross = cross_seed_policy_tv(
        bundles[0].policy, bundles[1].policy, observations, device=args.device
    )
    fit_pass = all(
        x["final_fit"]["advantage_gate_pass"] and x["final_fit"]["policy_gate_pass"]
        for x in reports
    )
    payload = {
        "schema": "SPINCORE_R7_3_ADVANTAGE_ENSEMBLE_END_TO_END_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "ensemble_size": int(args.ensemble_size),
        "algorithm_seeds": seeds,
        "shared_deck_stream_seed": int(args.deck_stream_seed),
        "iterations": int(args.iterations),
        "roots_per_iteration": int(args.roots_per_iteration),
        "per_seed": reports,
        "cross_seed": {k: float(v) for k, v in cross.items()},
        "observation_count": len(observations),
        "per_seed_fit_pass": bool(fit_pass),
        "frozen_gates": dict(FROZEN_GATES),
        "interpretation_note": (
            "Diagnostic only. Every CFR iteration trains N independent AdvantageNets from scratch "
            "on the exact same accumulated Advantage reservoir. Their raw predicted advantages "
            "are averaged before the unchanged production hard regret-matching map; AveragePolicy "
            "collection/training remains single-network. This tests whether same-memory fit "
            "ensembling suppresses upstream behavior variance end-to-end."
        ),
        "acceptance_gate_changed": False,
        "production_ensemble_changed": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
