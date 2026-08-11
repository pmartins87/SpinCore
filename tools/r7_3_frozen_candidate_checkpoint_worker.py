from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

import run_r7_3_partial_exact_ensemble_paired as base
import run_r7_3_partial_exact_policy_mixture_paired as policy_mixture
import run_r7_3_policy_mixture_temporal_blend as temporal
import run_r7_3_policy_mixture_uncertainty_damping as uncertainty
from spincore.deep_cfr import DeepCFRDomainSession, icm_delta_utility
from spincore.r7 import (
    MidIterationProgress,
    audit_model_fit,
    cross_seed_policy_tv,
    load_checkpoint,
    save_checkpoint,
)
from spincore.r7_candidate_checkpoint import (
    pack_candidate_behavior,
    restore_candidate_behavior_models,
)
from spincore.solver import SolverLibrary


FREEZE_SCHEMA = "SPINCORE_R7_3_CANDIDATE_SEMANTIC_FREEZE_V1"
REPORT_SCHEMA = "SPINCORE_R7_3_CANDIDATE_CHECKPOINT_RECERT_V1"
PAYOUT = (0.5, 0.3, 0.2)


def _args_from_freeze(freeze: dict) -> SimpleNamespace:
    ec = dict(freeze["execution_contract"])
    return SimpleNamespace(
        device="cpu",
        lr=1e-3,
        reservoir_capacity=int(ec["reservoir_capacity"]),
        advantage_chunk_steps=int(ec["advantage_chunk_steps"]),
        advantage_max_steps_per_iteration=int(ec["advantage_max_steps_per_iteration"]),
        advantage_fit_target=float(ec["advantage_fit_target"]),
        policy_chunk_steps=int(ec["policy_chunk_steps"]),
        policy_max_steps=int(ec["policy_max_steps"]),
        policy_fit_target=float(ec["policy_fit_target"]),
        batch_size=int(ec["batch_size"]),
        audit_size=int(ec["audit_size"]),
        cross_seed_per_seed=int(ec["cross_seed_per_seed"]),
        exact_opponent_levels=int(ec["exact_opponent_levels"]),
        roots_per_iteration=int(ec["roots_per_iteration"]),
        iterations=int(ec["iterations"]),
    )


def _configure_behavior(freeze: dict):
    kind = str(freeze["behavior_kind"])
    params = dict(freeze.get("params") or {})
    if kind == "uncertainty_damping":
        uncertainty.EPSILON_SCALE = float(params["epsilon_scale"])
        uncertainty.EPSILON_CAP = float(params["epsilon_cap"])
        return uncertainty.UncertaintyDampedPolicyMixture(device="cpu")
    if kind == "temporal_blend":
        temporal.CURRENT_WEIGHT = float(params["current_policy_weight"])
        return temporal.TemporalBlendPolicyMixture(device="cpu")
    if kind == "policy_mixture":
        return policy_mixture.PolicyMixtureEnsembleAdvantagePolicy(device="cpu")
    raise ValueError(f"unsupported behavior kind {kind!r}")


def _behavior_models(behavior) -> tuple[list, list, int]:
    if isinstance(behavior, temporal.TemporalBlendPolicyMixture):
        return list(behavior.current_models), list(behavior.previous_models), int(behavior.fit_generation)
    return list(behavior.models), [], 0


def _restore_behavior(freeze: dict, current: list, previous: list, fit_generation: int):
    behavior = _configure_behavior(freeze)
    if isinstance(behavior, temporal.TemporalBlendPolicyMixture):
        behavior.current_models = list(current)
        behavior.previous_models = list(previous)
        behavior.fit_generation = int(fit_generation)
    else:
        behavior.models = list(current)
    return behavior


def _runtime(bundle, behavior, solver: SolverLibrary):
    session = DeepCFRDomainSession(
        solver_library=solver,
        bundle=bundle,
        terminal_utility=icm_delta_utility(PAYOUT),
        device="cpu",
    )
    partial = base.PartialExactAdvantageCollector(
        policy=behavior,
        terminal_utility=session.terminal_utility,
        rng=bundle.batch_rng,
        advantage_memory=bundle.adv_mem,
        strategy_memory=bundle.pol_mem,
    )
    session.collector = partial
    return session, partial


def _new_seed(seed: int, freeze: dict, solver: SolverLibrary, args):
    bundle = base.diagnostic.make_bundle(
        int(seed),
        device=args.device,
        reservoir_capacity=int(args.reservoir_capacity),
        lr=float(args.lr),
    )
    behavior = _configure_behavior(freeze)
    session, partial = _runtime(bundle, behavior, solver)
    return bundle, behavior, session, partial


def _run_iteration(*, seed: int, iteration: int, bundle, behavior, session, partial, solver, args, ensemble_size: int):
    episode = base.diagnostic.hu_episode()
    live = [i for i, stack in enumerate(episode.stacks) if stack > 0]
    for root_index in range(int(args.roots_per_iteration)):
        ds = base.deck_seed(seed, iteration, root_index, int(args.roots_per_iteration))
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

    memory_state = bundle.adv_mem.state_dict()
    primary_model, _primary_report = base._fit_primary_member(
        session=session,
        bundle=bundle,
        seed=int(seed),
        iteration=int(iteration),
        args=args,
    )
    models = [primary_model]
    for member in range(1, int(ensemble_size)):
        model, _report = base._train_member(
            memory_state=memory_state,
            algorithm_seed=int(seed),
            iteration=int(iteration),
            member=int(member),
            solver=solver,
            args=args,
        )
        models.append(model)
    behavior.models = models


def _finish(*, seed: int, bundle, behavior, session, args):
    policy_progress = base._fit_policy(
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
    policy_audit = audit_model_fit(
        bundle,
        sample_size=max(int(args.audit_size), 2048),
        seed=int(seed) ^ 0x2468ACE0,
        device=args.device,
    )
    current, previous, fit_generation = _behavior_models(behavior)
    final_nrmse = base._ensemble_nrmse(
        current,
        bundle.adv_mem,
        sample_size=max(int(args.audit_size), 2048),
        seed=int(seed) ^ 0x13572468,
        device=args.device,
    )
    return {
        "policy_progress": policy_progress,
        "ensemble_advantage_weighted_nrmse": float(final_nrmse),
        "policy_weighted_mean_tv": float(policy_audit["policy_weighted_mean_tv"]),
        "current_member_count": len(current),
        "previous_member_count": len(previous),
        "fit_generation": int(fit_generation),
    }


def _eq(a: Any, b: Any) -> bool:
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and torch.equal(a.cpu(), b.cpu())
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return set(a) == set(b) and all(_eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b
    return a == b


def _model_states(models) -> list[dict]:
    return [{k: v.detach().cpu().clone() for k, v in model.state_dict().items()} for model in models]


def _snapshot(bundle, behavior, finish_report: dict, torch_rng: torch.Tensor) -> dict:
    current, previous, fit_generation = _behavior_models(behavior)
    return {
        "counters": dict(bundle.counters),
        "adv_mem": bundle.adv_mem.state_dict(),
        "pol_mem": bundle.pol_mem.state_dict(),
        "batch_rng": bundle.batch_rng.getstate(),
        "torch_rng": torch_rng.clone().cpu(),
        "advantage": {k: v.detach().cpu().clone() for k, v in bundle.advantage.state_dict().items()},
        "policy": {k: v.detach().cpu().clone() for k, v in bundle.policy.state_dict().items()},
        "adv_opt": bundle.adv_opt.state_dict(),
        "pol_opt": bundle.pol_opt.state_dict(),
        "current_models": _model_states(current),
        "previous_models": _model_states(previous),
        "fit_generation": int(fit_generation),
        "finish_report": finish_report,
    }


def _compare_snapshots(a: dict, b: dict) -> dict[str, bool]:
    return {key: _eq(a[key], b[key]) for key in a}


def _run_seed(*, seed: int, freeze: dict, solver: SolverLibrary, args, split_iteration: int, checkpoint_dir: Path):
    ensemble_size = int(freeze["ensemble_size"])
    bundle, behavior, session, partial = _new_seed(seed, freeze, solver, args)

    for iteration in range(1, split_iteration + 1):
        _run_iteration(
            seed=seed, iteration=iteration, bundle=bundle, behavior=behavior,
            session=session, partial=partial, solver=solver, args=args,
            ensemble_size=ensemble_size,
        )

    current, previous, fit_generation = _behavior_models(behavior)
    behavior_payload = pack_candidate_behavior(
        kind=str(freeze["behavior_kind"]),
        primary_model=bundle.advantage,
        current_models=current,
        previous_models=previous,
        params=dict(freeze.get("params") or {}),
        fit_generation=int(fit_generation or split_iteration),
    )
    checkpoint_path = checkpoint_dir / f"seed_{seed}.pt"
    save_checkpoint(
        checkpoint_path,
        bundle,
        MidIterationProgress(
            iteration=int(split_iteration),
            phase="post_advantage_fit",
            root_index=int(args.roots_per_iteration),
        ),
        {"candidate_behavior": behavior_payload, "semantic_freeze": freeze},
    )

    # Continuous branch.
    for iteration in range(split_iteration + 1, int(args.iterations) + 1):
        _run_iteration(
            seed=seed, iteration=iteration, bundle=bundle, behavior=behavior,
            session=session, partial=partial, solver=solver, args=args,
            ensemble_size=ensemble_size,
        )
    cont_finish = _finish(seed=seed, bundle=bundle, behavior=behavior, session=session, args=args)
    cont_torch_rng = torch.get_rng_state().clone()
    cont_snapshot = _snapshot(bundle, behavior, cont_finish, cont_torch_rng)

    # Restore branch. load_checkpoint restores base torch RNG; side-model restore
    # is required to be RNG-neutral by SPINCORE_R7_CANDIDATE_BEHAVIOR_V1.
    restored_bundle, progress, extra = load_checkpoint(checkpoint_path, device=args.device)
    if int(progress.iteration) != int(split_iteration) or progress.phase != "post_advantage_fit":
        raise RuntimeError("checkpoint progress mismatch")
    restored_current, restored_previous, meta = restore_candidate_behavior_models(
        extra["candidate_behavior"],
        config=restored_bundle.config,
        primary_model=restored_bundle.advantage,
        device=args.device,
    )
    restored_behavior = _restore_behavior(
        freeze,
        restored_current,
        restored_previous,
        int(meta.get("fit_generation") or split_iteration),
    )
    restored_session, restored_partial = _runtime(restored_bundle, restored_behavior, solver)

    for iteration in range(split_iteration + 1, int(args.iterations) + 1):
        _run_iteration(
            seed=seed, iteration=iteration, bundle=restored_bundle, behavior=restored_behavior,
            session=restored_session, partial=restored_partial, solver=solver, args=args,
            ensemble_size=ensemble_size,
        )
    restored_finish = _finish(
        seed=seed,
        bundle=restored_bundle,
        behavior=restored_behavior,
        session=restored_session,
        args=args,
    )
    restored_torch_rng = torch.get_rng_state().clone()
    restored_snapshot = _snapshot(restored_bundle, restored_behavior, restored_finish, restored_torch_rng)
    checks = _compare_snapshots(cont_snapshot, restored_snapshot)
    return {
        "seed": int(seed),
        "split_iteration": int(split_iteration),
        "checks": checks,
        "all_exact": all(checks.values()),
        "continuous_finish": cont_finish,
        "restored_finish": restored_finish,
    }, bundle, restored_bundle


def main() -> int:
    ap = argparse.ArgumentParser(description="Physical deterministic checkpoint/resume recertification for a frozen R7.3 winner")
    ap.add_argument("--freeze", type=Path, required=True)
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split-iteration", type=int, default=3)
    args_cli = ap.parse_args()

    freeze = json.loads(args_cli.freeze.read_text(encoding="utf-8"))
    if freeze.get("schema") != FREEZE_SCHEMA or freeze.get("evidence_r7_3_pass") is not True:
        raise SystemExit("requires a valid gate-clearing semantic freeze")
    args = _args_from_freeze(freeze)
    if not (1 <= int(args_cli.split_iteration) < int(args.iterations)):
        raise SystemExit("split-iteration must be inside the frozen training horizon")
    seeds = [int(x) for x in freeze.get("algorithm_seeds", [20260829, 20260807])]
    if len(seeds) != 2:
        raise SystemExit("checkpoint recertification requires exactly two frozen seeds")

    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    solver = SolverLibrary(args_cli.solver)
    args_cli.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    seed_reports = []
    continuous_bundles = []
    restored_bundles = []
    for seed in seeds:
        report, cont, restored = _run_seed(
            seed=seed,
            freeze=freeze,
            solver=solver,
            args=args,
            split_iteration=int(args_cli.split_iteration),
            checkpoint_dir=args_cli.checkpoint_dir,
        )
        seed_reports.append(report)
        continuous_bundles.append(cont)
        restored_bundles.append(restored)

    cont_obs = base.diagnostic.shared_cross_seed_observations(
        continuous_bundles,
        per_seed=int(args.cross_seed_per_seed),
        seed=0x715EED,
    )
    restored_obs = base.diagnostic.shared_cross_seed_observations(
        restored_bundles,
        per_seed=int(args.cross_seed_per_seed),
        seed=0x715EED,
    )
    corpus_exact = cont_obs == restored_obs
    cont_cross = cross_seed_policy_tv(
        continuous_bundles[0].policy,
        continuous_bundles[1].policy,
        cont_obs,
        device=args.device,
    )
    restored_cross = cross_seed_policy_tv(
        restored_bundles[0].policy,
        restored_bundles[1].policy,
        restored_obs,
        device=args.device,
    )
    cross_exact = _eq(cont_cross, restored_cross)
    passed = all(row["all_exact"] for row in seed_reports) and corpus_exact and cross_exact

    payload = {
        "schema": REPORT_SCHEMA,
        "label": freeze["label"],
        "behavior_semantic_id": freeze["behavior_semantic_id"],
        "source_head_sha": freeze["source_head_sha"],
        "evidence_commit_sha": freeze.get("evidence_commit_sha"),
        "algorithm_seeds": seeds,
        "split_iteration": int(args_cli.split_iteration),
        "frozen_iterations": int(args.iterations),
        "frozen_roots_per_iteration": int(args.roots_per_iteration),
        "per_seed": seed_reports,
        "cross_seed_observation_corpus_exact": bool(corpus_exact),
        "continuous_cross_seed": {k: float(v) for k, v in cont_cross.items()},
        "restored_cross_seed": {k: float(v) for k, v in restored_cross.items()},
        "cross_seed_metrics_exact": bool(cross_exact),
        "checkpoint_resume_recertification_pass": bool(passed),
        "acceptance_gate_changed": False,
        "ready_for_640": False,
        "ready_for_tables": False,
    }
    args_cli.out.parent.mkdir(parents=True, exist_ok=True)
    args_cli.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
