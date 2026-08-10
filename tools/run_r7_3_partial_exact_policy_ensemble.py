from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

import run_r7_3_diagnostic as diagnostic
from run_r7_3_partial_exact_advantage_screen import PartialExactAdvantageCollector
from run_r7_3_replicated_640_candidate import _fit_policy, _fit_pass
from run_r7_3_variance_decomposition import _advantage_fit_nrmse, _finite

from spincore.deep_cfr import DeepCFRDomainSession, _batch, icm_delta_utility
from spincore.r7 import (
    FROZEN_GATES,
    stratified_audit_indices,
    weighted_mean_tv,
)
from spincore.solver import SolverLibrary
from spincore_nn import AveragePolicyNet, UniformReservoir
from spincore_nn.codec import collate_inputs, decode_spnniv1
from spincore_nn.training import train_step


diagnostic.HISTORICAL_PARAMS_PER_NETWORK = 152_434
DEFAULT_SEEDS = (20260829, 20260807)
PAYOUT = (0.5, 0.3, 0.2)
POLICY_MEMBER_INIT_XOR = 0x50A1C1E
POLICY_MEMBER_BATCH_XOR = 0xA9E12C7


def deck_seed(seed: int, iteration: int, root_index: int, roots_per_iteration: int) -> int:
    global_root = (int(iteration) - 1) * int(roots_per_iteration) + int(root_index)
    return (int(seed) * 1_000_003 + global_root * 97 + int(iteration)) & ((1 << 64) - 1)


class AveragePolicyEnsemble:
    def __init__(self, models):
        self.models = list(models)
        if not self.models:
            raise ValueError("AveragePolicyEnsemble requires at least one model")

    def eval(self):
        for model in self.models:
            model.eval()
        return self

    def probabilities(self, batch):
        total = None
        for model in self.models:
            model.eval()
            p = model.probabilities(batch)
            total = p if total is None else total + p
        return total / float(len(self.models))


def _policy_fit_tv(model, memory, *, sample_size: int, seed: int, device: str) -> float:
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


def _train_side_policy_member(*, memory_state, algorithm_seed: int, member: int, config, args):
    init_seed = (
        int(algorithm_seed)
        ^ POLICY_MEMBER_INIT_XOR
        ^ (int(member) * 0x45D9F3B)
    ) & 0x7FFFFFFF
    batch_seed = (
        int(algorithm_seed)
        ^ POLICY_MEMBER_BATCH_XOR
        ^ (int(member) * 0xC2B2AE3D)
    ) & ((1 << 64) - 1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(init_seed))
        model = AveragePolicyNet(config).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    memory = UniformReservoir.from_state_dict(memory_state)
    rng = random.Random(int(batch_seed))
    local_steps = 0
    progress = []
    audit_seed = int(algorithm_seed) ^ 0x13579BDF ^ (int(member) * 0x2468AC)
    while local_steps < int(args.policy_max_steps):
        steps = min(int(args.policy_chunk_steps), int(args.policy_max_steps) - local_steps)
        losses = []
        for _ in range(steps):
            samples = memory.sample(min(int(args.batch_size), len(memory.items)), rng)
            batch, target, weights = _batch(samples, args.device)
            losses.append(float(train_step(model, optimizer, batch, target, weights, "strategy")))
        local_steps += steps
        tv = _policy_fit_tv(
            model,
            memory,
            sample_size=int(args.audit_size),
            seed=audit_seed,
            device=args.device,
        )
        row = {
            "optimizer_steps": int(local_steps),
            "weighted_mean_tv": float(tv),
            "mean_training_loss": sum(losses) / max(len(losses), 1),
            "frozen_gate_pass": _finite(tv)
            and float(tv) <= FROZEN_GATES["policy_weighted_mean_tv_max"],
            "fit_target_reached": _finite(tv)
            and float(tv) <= float(args.policy_fit_target),
        }
        progress.append(row)
        if row["fit_target_reached"]:
            break
    return model, {
        "member": int(member),
        "role": "SIDE_POLICY_MEMBER_DOES_NOT_PERTURB_PRIMARY_RNG",
        "init_seed": int(init_seed),
        "batch_seed": int(batch_seed),
        "optimizer_steps": int(local_steps),
        "final_weighted_mean_tv": float(progress[-1]["weighted_mean_tv"]),
        "progress": progress,
    }


def run_seed(*, seed: int, solver: SolverLibrary, args):
    bundle = diagnostic.make_bundle(
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
    partial = PartialExactAdvantageCollector(
        policy=session.behavior,
        terminal_utility=session.terminal_utility,
        rng=bundle.batch_rng,
        advantage_memory=bundle.adv_mem,
        strategy_memory=bundle.pol_mem,
    )
    session.collector = partial
    episode = diagnostic.hu_episode()
    live = [i for i, stack in enumerate(episode.stacks) if stack > 0]
    checkpoints = []

    for iteration in range(1, int(args.iterations) + 1):
        for root_index in range(int(args.roots_per_iteration)):
            ds = deck_seed(seed, iteration, root_index, int(args.roots_per_iteration))
            nodes = advantage_added = strategy_added = 0
            for traverser in live:
                root = solver.create(episode, int(ds))
                try:
                    result = partial.collect_advantage_partial_exact(
                        root,
                        traverser=int(traverser),
                        iteration=int(iteration),
                        exact_opponent_levels=int(args.exact_opponent_levels),
                    )
                finally:
                    root.close()
                nodes += int(result.nodes)
                advantage_added += int(result.samples_added)
            for target_player in live:
                root = solver.create(episode, int(ds))
                try:
                    strategy_added += int(
                        partial.collect_strategy_own_reach(
                            root,
                            target_player=int(target_player),
                            iteration=int(iteration),
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

        reset_seed = (int(seed) ^ (int(iteration) * 0x9E3779B1)) & 0x7FFFFFFF
        session.reset_advantage_network(init_seed=reset_seed, lr=float(args.lr))
        local_steps = 0
        progress = []
        audit_seed = int(seed) ^ (int(iteration) * 0x45D9F3B)
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
            row = {
                "optimizer_steps": int(local_steps),
                "weighted_nrmse": float(nrmse),
                "frozen_gate_pass": _fit_pass(nrmse, "advantage_weighted_nrmse_max"),
                "fit_target_reached": _finite(nrmse)
                and float(nrmse) <= float(args.advantage_fit_target),
            }
            progress.append(row)
            if row["fit_target_reached"]:
                break
        checkpoints.append(
            {
                "iteration": int(iteration),
                "roots": int(bundle.counters["roots"]),
                "nodes": int(bundle.counters["nodes"]),
                "advantage_samples": len(bundle.adv_mem.items),
                "advantage_seen": int(bundle.adv_mem.seen),
                "strategy_samples": len(bundle.pol_mem.items),
                "strategy_seen": int(bundle.pol_mem.seen),
                "final_advantage_fit": progress[-1],
            }
        )

    memory_state = bundle.pol_mem.state_dict()
    # Primary policy member exactly follows the authoritative live bundle RNG.
    primary_progress = _fit_policy(
        bundle=bundle,
        session=session,
        seed=int(seed),
        device=args.device,
        chunk_steps=int(args.policy_chunk_steps),
        max_steps=int(args.policy_max_steps),
        fit_target=float(args.policy_fit_target),
        batch_size=int(args.batch_size),
        audit_size=int(args.audit_size),
    )
    primary = bundle.policy
    members = [primary]
    member_reports = [
        {
            "member": 0,
            "role": "PRIMARY_AUTHORITATIVE_COUPLED_RNG",
            "optimizer_steps": int(bundle.counters["policy_optimizer_steps"]),
            "final_weighted_mean_tv": float(primary_progress[-1]["weighted_mean_tv"]),
            "progress": primary_progress,
        }
    ]
    for member in range(1, 4):
        model, report = _train_side_policy_member(
            memory_state=memory_state,
            algorithm_seed=int(seed),
            member=int(member),
            config=bundle.config,
            args=args,
        )
        members.append(model)
        member_reports.append(report)

    policies = {
        1: AveragePolicyEnsemble(members[:1]),
        2: AveragePolicyEnsemble(members[:2]),
        4: AveragePolicyEnsemble(members[:4]),
    }
    policy_fit = {
        str(size): _policy_fit_tv(
            policy,
            bundle.pol_mem,
            sample_size=max(int(args.audit_size), 2048),
            seed=int(seed) ^ 0x2468ACE0,
            device=args.device,
        )
        for size, policy in policies.items()
    }
    advantage_fit = _advantage_fit_nrmse(
        bundle,
        sample_size=max(int(args.audit_size), 2048),
        seed=int(seed) ^ 0x13572468,
        device=args.device,
    )
    report = {
        "algorithm_seed": int(seed),
        "roots": int(bundle.counters["roots"]),
        "nodes": int(bundle.counters["nodes"]),
        "advantage_samples": len(bundle.adv_mem.items),
        "advantage_seen": int(bundle.adv_mem.seen),
        "strategy_samples": len(bundle.pol_mem.items),
        "strategy_seen": int(bundle.pol_mem.seen),
        "checkpoints": checkpoints,
        "advantage_weighted_nrmse": float(advantage_fit),
        "advantage_gate_pass": bool(
            advantage_fit <= FROZEN_GATES["advantage_weighted_nrmse_max"]
        ),
        "policy_members": member_reports,
        "policy_ensemble_weighted_mean_tv": {k: float(v) for k, v in policy_fit.items()},
        "policy_ensemble_gate_pass": {
            k: bool(v <= FROZEN_GATES["policy_weighted_mean_tv_max"])
            for k, v in policy_fit.items()
        },
    }
    return bundle, policies, report


def _cross_policy_tv(model_a, model_b, observations, *, device: str):
    if not observations:
        return {"mean_tv": math.inf, "p50_tv": math.inf, "p95_tv": math.inf, "max_tv": math.inf}
    batch = collate_inputs([decode_spnniv1(x) for x in observations], device=device)
    model_a.eval(); model_b.eval()
    with torch.no_grad():
        a = model_a.probabilities(batch).detach().cpu()
        b = model_b.probabilities(batch).detach().cpu()
    tv = 0.5 * torch.abs(a - b).sum(1)
    q = torch.quantile(tv, torch.tensor([0.5, 0.95]))
    return {
        "mean_tv": float(tv.mean()),
        "p50_tv": float(q[0]),
        "p95_tv": float(q[1]),
        "max_tv": float(tv.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Authoritative partial-exact CFR plus final AveragePolicy ensemble screen")
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    ap.add_argument("--exact-opponent-levels", type=int, default=2)
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
    if len(seeds) != 2 or int(args.exact_opponent_levels) <= 0:
        raise SystemExit("requires exactly two seeds and positive exact-opponent level")
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    solver = SolverLibrary(args.solver)
    started = time.time()
    bundles = []
    policy_sets = []
    reports = []
    for seed in seeds:
        bundle, policies, report = run_seed(seed=int(seed), solver=solver, args=args)
        bundles.append(bundle)
        policy_sets.append(policies)
        reports.append(report)

    observations = diagnostic.shared_cross_seed_observations(
        bundles, per_seed=int(args.cross_seed_per_seed), seed=0x715EED
    )
    cross = {
        str(size): _cross_policy_tv(
            policy_sets[0][size], policy_sets[1][size], observations, device=args.device
        )
        for size in (1, 2, 4)
    }
    pass_by_size = {}
    for size in (1, 2, 4):
        k = str(size)
        fit_pass = all(
            row["advantage_gate_pass"] and row["policy_ensemble_gate_pass"][k]
            for row in reports
        )
        cross_pass = (
            _finite(cross[k]["mean_tv"])
            and _finite(cross[k]["p95_tv"])
            and cross[k]["mean_tv"] <= FROZEN_GATES["cross_seed_mean_tv_max"]
            and cross[k]["p95_tv"] <= FROZEN_GATES["cross_seed_p95_tv_max"]
        )
        pass_by_size[k] = {
            "fit_pass": bool(fit_pass),
            "cross_seed_pass": bool(cross_pass),
            "r7_3_pass": bool(fit_pass and cross_pass),
        }

    baseline_mean = max(cross["1"]["mean_tv"], 1e-12)
    baseline_p95 = max(cross["1"]["p95_tv"], 1e-12)
    payload = {
        "schema": "SPINCORE_R7_3_PARTIAL_EXACT_POLICY_ENSEMBLE_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "algorithm_seeds": seeds,
        "exact_opponent_levels": int(args.exact_opponent_levels),
        "iterations": int(args.iterations),
        "roots_per_iteration": int(args.roots_per_iteration),
        "roots_per_seed": int(args.iterations * args.roots_per_iteration),
        "deck_semantics": "GENERATION2_AUTHORITATIVE_GLOBAL_ROOT_FORMULA_EXACT",
        "deck_formula": "seed*1000003 + global_root*97 + iteration",
        "rng_contract": "RECOVERED_SINGLE_COUPLED_BATCH_RNG_PRIMARY_POLICY_MEMBER",
        "extra_policy_members_perturb_primary_rng": False,
        "per_seed": reports,
        "cross_seed_observation_count": len(observations),
        "cross_seed_by_policy_ensemble_size": cross,
        "pass_by_policy_ensemble_size": pass_by_size,
        "ratios_to_size1": {
            k: {
                "mean_tv_ratio": float(cross[k]["mean_tv"] / baseline_mean),
                "p95_tv_ratio": float(cross[k]["p95_tv"] / baseline_p95),
            }
            for k in ("1", "2", "4")
        },
        "frozen_gates": dict(FROZEN_GATES),
        "acceptance_gate_changed": False,
        "production_estimator_changed": False,
        "production_average_policy_ensemble_changed": False,
        "interpretation_note": (
            "Diagnostic only. CFR collection and Advantage fitting use the authoritative partial-exact "
            "level-2 deck/RNG contract. After CFR is frozen, member zero follows the recovered live "
            "AveragePolicy training stream and side policy members train on the same frozen strategy "
            "memory without perturbing that stream. Probability averaging directly tests whether "
            "final-policy approximation/extrapolation variance is responsible for the stubborn p95 tail."
        ),
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
