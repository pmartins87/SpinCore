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
from run_r7_3_partial_exact_640_candidate import deck_seed
from run_r7_3_replicated_640_candidate import _fit_policy, _fit_pass
from run_r7_3_variance_decomposition import _advantage_fit_nrmse, _finite

from spincore.deep_cfr import DeepCFRDomainSession, _batch, icm_delta_utility, regret_matching_policy, uniform_policy
from spincore.r7 import FROZEN_GATES, audit_model_fit, cross_seed_policy_tv, stratified_audit_indices, weighted_mean_tv
from spincore.solver import SolverLibrary
from spincore_nn import AveragePolicyNet, UniformReservoir
from spincore_nn.codec import collate_inputs, decode_spnniv1
from spincore_nn.reservoir import StrategySample
from spincore_nn.training import train_step


diagnostic.HISTORICAL_PARAMS_PER_NETWORK = 152_434
DEFAULT_SEEDS = (20260829, 20260807)
PAYOUT = (0.5, 0.3, 0.2)
SURROGATE_INIT_XOR = 0xD1AEC7B3
SURROGATE_BATCH_XOR = 0x5A7706A7
BASELINE_REFERENCE = {
    "file": "validation/R7_3_PARTIAL_EXACT_ENSEMBLE_PAIRED_SIZE1_256.json",
    "evidence_commit": "44ed648eec64c030cd7dbde40117b85fd19c0f8c",
    "roots_per_seed": 256,
    "cross_seed_mean_tv": 0.24565602838993073,
    "cross_seed_p95_tv": 0.6287055611610413,
    "per_seed_fit_pass": True,
}


class DirectBehaviorPolicy:
    """Experimental smooth behavior surrogate for diagnostic CFR collection.

    Before the first fit it is exactly zero-regret uniform.  Thereafter it emits
    probabilities from a freshly fitted AveragePolicyNet whose targets are
    sample-level hard-regret-matching policies derived from Advantage memory.
    This is intentionally *not* claimed equivalent to Deep CFR because RM is
    nonlinear.  It is a causal screen for the demonstrated sign/support noise.
    """

    def __init__(self, *, device: str):
        self.model = None
        self.device = device

    @property
    def ready(self) -> bool:
        return self.model is not None

    def __call__(self, state, observation: bytes, legal: tuple[int, ...]):
        if self.model is None:
            return uniform_policy(state, observation, legal)
        batch = collate_inputs([decode_spnniv1(observation)], device=self.device)
        self.model.eval()
        with torch.no_grad():
            probs = self.model.probabilities(batch)[0].detach().cpu().tolist()
        out = [0.0] * 6
        total = 0.0
        for action in legal:
            value = max(0.0, float(probs[action]))
            out[action] = value
            total += value
        if total <= 0.0:
            return uniform_policy(state, observation, legal)
        for action in legal:
            out[action] /= total
        return tuple(out)


def _surrogate_memory(adv_memory):
    memory = UniformReservoir(max(len(adv_memory.items), 1), 0xD1AEC7)
    memory.items = []
    for sample in adv_memory.items:
        legal_actions = tuple(i for i, yes in enumerate(sample.legal) if yes)
        target = tuple(regret_matching_policy(sample.target, legal_actions))
        memory.items.append(
            StrategySample(
                sample.observation,
                sample.legal,
                target,
                float(sample.weight),
                int(sample.iteration),
            )
        )
    memory.seen = len(memory.items)
    return memory


def _surrogate_fit_tv(model, memory, *, sample_size: int, seed: int, device: str) -> float:
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


def _fit_surrogate(*, bundle, algorithm_seed: int, iteration: int, args):
    memory = _surrogate_memory(bundle.adv_mem)
    init_seed = (
        int(algorithm_seed)
        ^ SURROGATE_INIT_XOR
        ^ (int(iteration) * 0x9E3779B1)
    ) & 0x7FFFFFFF
    batch_seed = (
        int(algorithm_seed)
        ^ SURROGATE_BATCH_XOR
        ^ (int(iteration) * 0x85EBCA77)
    ) & ((1 << 64) - 1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(init_seed))
        model = AveragePolicyNet(bundle.config).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    rng = random.Random(int(batch_seed))
    audit_seed = int(algorithm_seed) ^ (int(iteration) * 0x45D9F3B) ^ 0x51A77E
    local_steps = 0
    progress = []
    while local_steps < int(args.surrogate_max_steps_per_iteration):
        chunk = min(
            int(args.surrogate_chunk_steps),
            int(args.surrogate_max_steps_per_iteration) - local_steps,
        )
        losses = []
        for _ in range(chunk):
            samples = memory.sample(min(int(args.batch_size), len(memory.items)), rng)
            batch, target, weights = _batch(samples, args.device)
            losses.append(float(train_step(model, optimizer, batch, target, weights, "strategy")))
        local_steps += chunk
        tv = _surrogate_fit_tv(
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
            "fit_target_reached": _finite(tv) and float(tv) <= float(args.surrogate_fit_target),
            "reference_policy_fit_gate_pass": _finite(tv)
            and float(tv) <= FROZEN_GATES["policy_weighted_mean_tv_max"],
        }
        progress.append(row)
        if row["fit_target_reached"]:
            break
    return model, {
        "init_seed": int(init_seed),
        "batch_seed": int(batch_seed),
        "optimizer_steps": int(local_steps),
        "memory_items": len(memory.items),
        "final_weighted_mean_tv": float(progress[-1]["weighted_mean_tv"]),
        "reference_policy_fit_gate_pass": bool(progress[-1]["reference_policy_fit_gate_pass"]),
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
    behavior = DirectBehaviorPolicy(device=args.device)
    partial = PartialExactAdvantageCollector(
        policy=behavior,
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
        behavior_ready_at_collection_start = bool(behavior.ready)
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

        # Keep authoritative Advantage fitting and RNG consumption intact for the
        # frozen fit gate, even though this experimental mode does not use the
        # fitted AdvantageNet as its next-iteration behavior policy.
        reset_seed = (int(seed) ^ (int(iteration) * 0x9E3779B1)) & 0x7FFFFFFF
        session.reset_advantage_network(init_seed=reset_seed, lr=float(args.lr))
        local_steps = 0
        adv_progress = []
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
            adv_progress.append(row)
            if row["fit_target_reached"]:
                break

        surrogate_model, surrogate_report = _fit_surrogate(
            bundle=bundle,
            algorithm_seed=int(seed),
            iteration=int(iteration),
            args=args,
        )
        behavior.model = surrogate_model
        checkpoints.append(
            {
                "iteration": int(iteration),
                "roots": int(bundle.counters["roots"]),
                "nodes": int(bundle.counters["nodes"]),
                "advantage_samples": len(bundle.adv_mem.items),
                "advantage_seen": int(bundle.adv_mem.seen),
                "strategy_samples": len(bundle.pol_mem.items),
                "strategy_seen": int(bundle.pol_mem.seen),
                "behavior_ready_at_collection_start": behavior_ready_at_collection_start,
                "final_advantage_fit": adv_progress[-1],
                "surrogate_fit": surrogate_report,
            }
        )
        print(json.dumps({"seed": seed, "checkpoint": checkpoints[-1]}, sort_keys=True), flush=True)

    policy_progress = _fit_policy(
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
    final_fit = audit_model_fit(
        bundle,
        sample_size=max(int(args.audit_size), 2048),
        seed=int(seed) ^ 0x2468ACE0,
        device=args.device,
    )
    return bundle, {
        "algorithm_seed": int(seed),
        "roots": int(bundle.counters["roots"]),
        "nodes": int(bundle.counters["nodes"]),
        "advantage_samples": len(bundle.adv_mem.items),
        "advantage_seen": int(bundle.adv_mem.seen),
        "strategy_samples": len(bundle.pol_mem.items),
        "strategy_seen": int(bundle.pol_mem.seen),
        "advantage_optimizer_steps": int(bundle.counters["adv_optimizer_steps"]),
        "policy_optimizer_steps": int(bundle.counters["policy_optimizer_steps"]),
        "checkpoints": checkpoints,
        "policy_progress": policy_progress,
        "final_fit": {
            "advantage_weighted_nrmse": float(final_fit["advantage_weighted_nrmse"]),
            "policy_weighted_mean_tv": float(final_fit["policy_weighted_mean_tv"]),
            "advantage_gate_pass": _fit_pass(
                final_fit["advantage_weighted_nrmse"], "advantage_weighted_nrmse_max"
            ),
            "policy_gate_pass": _fit_pass(
                final_fit["policy_weighted_mean_tv"], "policy_weighted_mean_tv_max"
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Authoritative partial-exact E2E direct-behavior surrogate candidate")
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument("--out", type=Path, default=Path("validation/R7_3_DIRECT_BEHAVIOR_E2E_256.json"))
    ap.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    ap.add_argument("--exact-opponent-levels", type=int, default=2)
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--roots-per-iteration", type=int, default=128)
    ap.add_argument("--advantage-chunk-steps", type=int, default=256)
    ap.add_argument("--advantage-max-steps-per-iteration", type=int, default=4096)
    ap.add_argument("--advantage-fit-target", type=float, default=0.50)
    ap.add_argument("--surrogate-chunk-steps", type=int, default=256)
    ap.add_argument("--surrogate-max-steps-per-iteration", type=int, default=4096)
    ap.add_argument("--surrogate-fit-target", type=float, default=0.105)
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
        raise SystemExit("requires exactly two seeds")
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    solver = SolverLibrary(args.solver)
    started = time.time()
    bundles = []
    reports = []
    for seed in seeds:
        bundle, report = run_seed(seed=int(seed), solver=solver, args=args)
        bundles.append(bundle)
        reports.append(report)

    observations = diagnostic.shared_cross_seed_observations(
        bundles,
        per_seed=int(args.cross_seed_per_seed),
        seed=0x715EED,
    )
    cross = cross_seed_policy_tv(
        bundles[0].policy,
        bundles[1].policy,
        observations,
        device=args.device,
    )
    fit_pass = all(
        row["final_fit"]["advantage_gate_pass"] and row["final_fit"]["policy_gate_pass"]
        for row in reports
    )
    surrogate_reference_fit_pass = all(
        cp["surrogate_fit"]["reference_policy_fit_gate_pass"]
        for row in reports
        for cp in row["checkpoints"]
    )
    cross_pass = bool(
        _finite(cross["mean_tv"])
        and _finite(cross["p95_tv"])
        and float(cross["mean_tv"]) <= FROZEN_GATES["cross_seed_mean_tv_max"]
        and float(cross["p95_tv"]) <= FROZEN_GATES["cross_seed_p95_tv_max"]
    )
    mean_ratio = float(cross["mean_tv"]) / BASELINE_REFERENCE["cross_seed_mean_tv"]
    p95_ratio = float(cross["p95_tv"]) / BASELINE_REFERENCE["cross_seed_p95_tv"]
    payload = {
        "schema": "SPINCORE_R7_3_DIRECT_BEHAVIOR_E2E_V1",
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
        "primary_rng_contract": "RECOVERED_SINGLE_COUPLED_BATCH_RNG",
        "surrogate_training_perturbs_primary_rng": False,
        "behavior_semantics": "DIRECT_POLICY_SURROGATE_OF_SAMPLE_LEVEL_REGRET_MATCHED_ADVANTAGE_TARGETS",
        "baseline_reference": dict(BASELINE_REFERENCE),
        "per_seed": reports,
        "cross_seed_observation_count": len(observations),
        "cross_seed": {k: float(v) for k, v in cross.items()},
        "summary": {
            "candidate_to_baseline_mean_tv_ratio": float(mean_ratio),
            "candidate_to_baseline_p95_tv_ratio": float(p95_ratio),
            "surrogate_reference_fit_pass": bool(surrogate_reference_fit_pass),
            "diagnosis": (
                "DIRECT_BEHAVIOR_SURROGATE_MATERIAL_END_TO_END"
                if fit_pass and surrogate_reference_fit_pass and min(mean_ratio, p95_ratio) <= 0.75
                else "DIRECT_BEHAVIOR_SURROGATE_NOT_MATERIAL_END_TO_END"
            ),
        },
        "frozen_gates": dict(FROZEN_GATES),
        "per_seed_fit_pass": bool(fit_pass),
        "cross_seed_pass": bool(cross_pass),
        "r7_3_pass": bool(fit_pass and cross_pass),
        "acceptance_gate_changed": False,
        "production_algorithm_changed": False,
        "theoretical_equivalence_claimed": False,
        "interpretation_note": (
            "Experimental causal screen only. Authoritative partial-exact collection, deck schedule, "
            "Advantage fit gate and live primary batch RNG are preserved. Surrogate training uses an "
            "isolated side RNG and therefore does not consume the primary stream. The candidate changes "
            "only next-iteration behavior to a smooth policy network trained on sample-level "
            "regret-matched Advantage targets. Because regret matching is nonlinear, this is not claimed "
            "equivalent to Deep CFR and cannot be promoted without a separate algorithm/versioning review."
        ),
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
