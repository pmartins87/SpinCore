from __future__ import annotations

"""Mechanical runtime correction for the frozen R7.5.3C x4 remediation.

The original x4 worker deliberately collects four 64-root chunks and then calls
``run_one_phase2_v3_iteration`` with ``roots_per_iteration=0`` so that the
unchanged Phase-2 Advantage fit executes exactly once after all 256 roots.
The frozen stage function completed that fit but then divided its reporting
counters by ``roots_added == 0``.  No x4 admission metric depends on the
zero-root placeholder report: the base x4 worker immediately replaces the
root/tree fields with the accumulated four-chunk values.

This wrapper changes only that mechanical reporting path.  It executes the
same frozen Advantage fitting/audit code and updates the same persistent stage
state, but represents the temporary zero-root reporting rates as 0.0 until the
base x4 worker patches them with the true 256-root values.

For recovery of iteration-1 chunk-4 only, an optional
``SPINCORE_X4_RESUME_EXECUTION_SHA`` lets the corrected run consume the already
completed chunk-3 checkpoint from the failed source execution.  The new output
checkpoint is still written under the corrected run's own execution SHA.
"""

import os
import time

import spincore.r7_5_representation_v3_stage as stage
import r7_5_3c_chance_coverage_x4_domain_worker as base


def _fit_only_iteration(*, bundle, session, behavior, state: dict, config, target_iteration: int) -> dict:
    """Frozen Phase-2 Advantage fit with no additional root collection.

    This is a line-for-line semantic extraction of the fit/audit/state-update
    portion of ``stage.run_one_phase2_v3_iteration``.  The only intentional
    difference is that root-normalized placeholder fields are 0.0 because this
    helper is legal only with ``config.roots_per_iteration == 0``.  The calling
    x4 worker overwrites those fields with the real four-chunk totals.
    """
    if int(config.roots_per_iteration) != 0:
        raise ValueError("runtime-fix helper is legal only for zero-root fit calls")
    expected = int(state["completed_iteration"]) + 1
    if int(target_iteration) != expected:
        raise ValueError(f"Phase 2 must advance exactly one iteration: expected {expected}")
    if not 1 <= int(target_iteration) <= config.total_iterations:
        raise ValueError("Phase 2 target iteration out of range")

    global_root = int(state["global_root"])
    scenario_counts = list(state["scenario_counts"])
    adv_seen_before = int(bundle.adv_mem.seen)
    pol_seen_before = int(bundle.pol_mem.seen)

    fit_started = time.perf_counter()
    reset_seed = stage.primary_reset_seed(int(state["training_seed"]), int(target_iteration))
    session.reset_advantage_network(init_seed=reset_seed, lr=config.learning_rate)
    session.train_advantage(steps=config.advantage_steps, batch_size=config.batch_size)
    primary_nrmse = stage.audit_v3_advantage_model(
        bundle.advantage,
        bundle.adv_mem.items,
        representation=str(state["representation"]),
        sample_size=config.audit_size,
        seed=int(state["training_seed"]) ^ (int(target_iteration) * 0x45D9F3B),
    )
    models = [bundle.advantage]
    member_reports = [{
        "member": 0,
        "role": "PRIMARY_AUTHORITATIVE_COUPLED_RNG",
        "init_seed": int(reset_seed),
        "optimizer_steps": config.advantage_steps,
        "final_weighted_nrmse": float(primary_nrmse),
    }]
    for member in (1, 2, 3):
        init_seed, batch_seed = stage.side_member_seeds(
            int(state["training_seed"]), int(target_iteration), member
        )
        member_started = time.perf_counter()
        model, fit_report = stage.fit_independent_v3_advantage_member(
            bundle.adv_mem.items,
            representation=str(state["representation"]),
            init_seed=init_seed,
            batch_seed=batch_seed,
            steps=config.advantage_steps,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
        )
        nrmse = stage.audit_v3_advantage_model(
            model,
            bundle.adv_mem.items,
            representation=str(state["representation"]),
            sample_size=config.audit_size,
            seed=int(state["training_seed"]) ^ (int(target_iteration) * 0x13579B) ^ (member * 0x2468AC),
        )
        member_reports.append({
            **fit_report,
            "member": member,
            "role": "SIDE_MEMBER_DOES_NOT_PERTURB_PRIMARY_RNG",
            "final_weighted_nrmse": float(nrmse),
            "fit_seconds": float(time.perf_counter() - member_started),
        })
        models.append(model)
    behavior.models = models
    ensemble_nrmse = stage.ensemble_v3_advantage_nrmse(
        models,
        bundle.adv_mem.items,
        representation=str(state["representation"]),
        sample_size=config.audit_size,
        seed=int(state["training_seed"]) ^ (int(target_iteration) * 0x5EEDBEEF),
    )
    fit_seconds = time.perf_counter() - fit_started
    regret_proxy = stage._mean_positive_regret_proxy(
        bundle.adv_mem.items,
        sample_size=config.audit_size,
        seed=int(state["training_seed"]) ^ (int(target_iteration) * 0x27D4EB2D),
    )

    adv_seen_added = int(bundle.adv_mem.seen) - adv_seen_before
    pol_seen_added = int(bundle.pol_mem.seen) - pol_seen_before
    report = {
        "iteration": int(target_iteration),
        "roots_added": 0,
        "nodes_added": 0,
        "nodes_per_root": 0.0,
        "tree_collection_seconds": 0.0,
        "tree_seconds_per_root": 0.0,
        "advantage_fit_seconds": float(fit_seconds),
        "advantage_seen_added": adv_seen_added,
        "strategy_seen_added": pol_seen_added,
        "advantage_samples_per_root": 0.0,
        "strategy_samples_per_root": 0.0,
        "branch_geometry": session.collector.telemetry_snapshot(),
        "regret_proxy": regret_proxy,
        "ensemble_weighted_nrmse": float(ensemble_nrmse),
        "ensemble_advantage_gate_pass": bool(ensemble_nrmse <= stage.ADVANTAGE_NRMSE_MAX),
        "members": member_reports,
        "behavior_stats_after_fit": behavior.stats(),
        "peak_rss_bytes": stage._peak_rss_bytes(),
        "runtime_fix_zero_root_fit": True,
    }
    state["completed_iteration"] = int(target_iteration)
    state["global_root"] = global_root
    state["scenario_counts"] = scenario_counts
    state["iteration_reports"] = list(state["iteration_reports"]) + [report]
    state["advantage_fit_seconds_total"] = float(state["advantage_fit_seconds_total"]) + fit_seconds
    return report


def _install_source_checkpoint_compatibility() -> None:
    source_sha = os.environ.get("SPINCORE_X4_RESUME_EXECUTION_SHA", "").strip()
    if not source_sha:
        return
    original = base.load_representation_v3_checkpoint

    def _load_from_source(*args, **kwargs):
        requested = str(kwargs.get("expected_execution_sha") or "")
        if not requested:
            raise RuntimeError("corrected partial recovery requires expected execution SHA")
        kwargs["expected_execution_sha"] = source_sha
        return original(*args, **kwargs)

    base.load_representation_v3_checkpoint = _load_from_source
    print(f"X4_RUNTIME_FIX source_partial_execution_sha={source_sha}", flush=True)


def main() -> int:
    base.run_one_phase2_v3_iteration = _fit_only_iteration
    _install_source_checkpoint_compatibility()
    print("X4_RUNTIME_FIX zero_root_fit_reporting_guard=ACTIVE", flush=True)
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
