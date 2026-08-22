from __future__ import annotations

"""Mechanical execution guard for the frozen R7.5.3D Phase 2A ablation.

The Phase2A base runner deliberately reuses the admitted x4 pattern: collect four
64-root chunks, then execute exactly one frozen Advantage fit with a temporary
zero-root config.  The shared stage function still divides temporary reporting
fields by zero, so use the already-audited x4 fit-only helper that preserves the
same Advantage fit/audit/state semantics and emits zero placeholders until the
Phase2A runner patches them with the true 256-root totals.

This wrapper also pins the local AveragePolicy audit to the authoritative Phase-2
seed `training_seed ^ 0x71A5BEEF` for every capacity arm, makes the final
stage-report marker recoverable if power is lost after the atomic resume
checkpoint but before the small JSON report is written, guarantees that parallel
child seed workers execute this same guarded entrypoint, and parallelizes only the
three already-independent final Strategy-capacity policy-fit arms in isolated
processes.  COMMON/NATIVE fits remain sequential inside each arm process so the
memory is loaded once and each learner keeps its exact frozen RNG semantics.

No scientific dimension, training seed, chance schedule, model, threshold,
reservoir capacity arm, learner budget, Strategy sample, or traversal behavior is
changed here.
"""

import json
from pathlib import Path

import r7_5_3d_v1plus_phase2a_policy_fit_worker as policy_worker
import r7_5_3d_v1plus_phase2a_strategy_capacity as base
from r7_5_3c_chance_coverage_x4_domain_worker_runtimefix import _fit_only_iteration


def _validate_stream_prefix_recoverable(seed_root: Path, stage_index: int) -> None:
    for index in range(1, int(stage_index) + 1):
        sp = base._stream_path(seed_root, index)
        rp = base._report_path(seed_root, index)
        if not sp.is_file():
            raise RuntimeError(f"Phase2A completed Strategy stream missing at stage {index}: {sp}")
        if not rp.is_file():
            if index == int(stage_index):
                # The resume checkpoint is the authoritative atomic state. The
                # base runner rewrites this last small report from checkpoint
                # metadata immediately after this validation returns.
                continue
            raise RuntimeError(f"Phase2A completed stage report missing at stage {index}: {rp}")
        report = json.loads(rp.read_text(encoding="utf-8"))
        if int(report.get("stage_index", -1)) != index:
            raise RuntimeError("Phase2A stage-report identity mismatch")
        items = base.torch.load(sp, map_location="cpu", weights_only=False)
        if len(items) != int(report.get("strategy_stream_count", -1)):
            raise RuntimeError("Phase2A Strategy stream count mismatch")


def _all_arm_rows_complete(seed_root: Path, training_seed: int, arm_name: str) -> tuple[bool, dict]:
    policy_root = seed_root / "policies"
    rows = {}
    for mode in ("COMMON_LEARNER", "NATIVE_LEARNER"):
        key = f"{mode}__{arm_name}"
        artifact = policy_root / f"{key}.pt"
        meta = policy_root / f"{key}.json"
        saved = policy_worker._valid_existing(
            meta,
            artifact,
            training_seed=int(training_seed),
            arm=str(arm_name),
        )
        if saved is None:
            return False, rows
        rows[key] = saved
    return True, rows


def _fit_seed_policies_parallel(*, seed_root: Path, training_seed: int, bundle, arms):
    """Fit the three passive capacity arms in isolated processes.

    The parent materializes each already-built reservoir state exactly once to a
    temporary context.  Child processes do no traversal and cannot mutate the
    authoritative bundle.  Each child owns one capacity arm and fits COMMON then
    NATIVE sequentially with the same seeds/budgets as the frozen experiment.
    """
    policy_root = seed_root / "policies"
    policy_root.mkdir(parents=True, exist_ok=True)
    context_root = seed_root / "fit_context"
    context_root.mkdir(parents=True, exist_ok=True)
    native_state = bundle.batch_rng.getstate()
    pending = []
    rows = {}

    for arm_name in base.CAPACITIES:
        complete, existing = _all_arm_rows_complete(seed_root, int(training_seed), arm_name)
        rows.update(existing)
        if complete:
            print(f"[Phase2A policy arm resume] seed={training_seed} {arm_name} complete", flush=True)
            continue
        context_path = context_root / f"{arm_name}.pt"
        context = {
            "schema": policy_worker.CONTEXT_SCHEMA,
            "training_seed": int(training_seed),
            "arm": str(arm_name),
            "capacity": int(base.CAPACITIES[arm_name]),
            "memory_state": arms[arm_name].state_dict(),
            "native_batch_rng_state": native_state,
            "production_training_authorized": False,
            "ready_for_tables": False,
        }
        base._atomic_torch_save(context, context_path)
        cmd = [
            base.sys.executable,
            str(Path(policy_worker.__file__).resolve()),
            "--context", str(context_path.resolve()),
            "--seed-root", str(seed_root.resolve()),
            "--training-seed", str(int(training_seed)),
            "--arm", str(arm_name),
        ]
        pending.append((arm_name, context_path, cmd))

    if pending:
        workers = min(3, len(pending))
        print(
            f"[Phase2A policy parallel] seed={training_seed} arms={len(pending)} "
            f"processes={workers} torch_threads_per_process={base.TORCH_THREADS}",
            flush=True,
        )
        with base.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(base.subprocess.run, cmd, check=False): (arm_name, context_path)
                for arm_name, context_path, cmd in pending
            }
            for future in base.as_completed(futures):
                arm_name, context_path = futures[future]
                completed = future.result()
                if int(completed.returncode) != 0:
                    raise RuntimeError(
                        f"Phase2A policy-fit arm worker {training_seed}/{arm_name} "
                        f"failed with exit code {completed.returncode}; preserve {context_path}"
                    )

    # Re-read authoritative metadata after all workers complete.  Only now remove
    # the large temporary contexts so interrupted runs remain recoverable.
    rows = {}
    for arm_name in base.CAPACITIES:
        complete, existing = _all_arm_rows_complete(seed_root, int(training_seed), arm_name)
        if not complete:
            raise RuntimeError(f"Phase2A policy arm incomplete after parallel fit: {training_seed}/{arm_name}")
        rows.update(existing)
    for _arm_name, context_path, _cmd in pending:
        try:
            context_path.unlink()
        except FileNotFoundError:
            pass
    try:
        context_root.rmdir()
    except OSError:
        pass
    return rows


def _run_parent_guarded(args) -> int:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    guarded_entrypoint = str(Path(__file__).resolve())
    commands = []
    for seed in base.TRAINING_SEEDS:
        cmd = [
            base.sys.executable,
            guarded_entrypoint,
            "--repo-root", str(Path(args.repo_root).resolve()),
            "--solver", str(Path(args.solver).resolve()),
            "--heldout-root", str(Path(args.heldout_root).resolve()),
            "--output-root", str(output_root),
            "--execution-sha", str(args.execution_sha),
            "--single-seed", str(int(seed)),
        ]
        commands.append((int(seed), cmd))
    with base.ThreadPoolExecutor(max_workers=min(int(args.seed_workers), len(commands))) as pool:
        futures = {pool.submit(base.subprocess.run, cmd, check=False): seed for seed, cmd in commands}
        for future in base.as_completed(futures):
            seed = futures[future]
            completed = future.result()
            if int(completed.returncode) != 0:
                raise RuntimeError(f"Phase2A guarded seed worker {seed} failed with exit code {completed.returncode}")
    result = base._evaluate_parent(args)
    out = output_root / "R7_5_3D_V1PLUS_PHASE2A_RESULT.json"
    base._atomic_json(result, out)
    print(json.dumps({
        "status": result["status"],
        "common_mean_tv": result["pooled_mean_tv"]["COMMON_LEARNER"],
        "native_mean_tv": result["pooled_mean_tv"]["NATIVE_LEARNER"],
        "absolute_improvement_100k_to_800k": result["decision"]["common_100k_to_800k_absolute_improvement"],
        "relative_improvement_100k_to_800k": result["decision"]["common_100k_to_800k_relative_improvement"],
        "result": str(out),
    }, indent=2, sort_keys=True), flush=True)
    return 0


def main() -> int:
    base.run_one_phase2_v3_iteration = _fit_only_iteration
    base._validate_stream_prefix = _validate_stream_prefix_recoverable
    base._fit_seed_policies = _fit_seed_policies_parallel
    base._run_parent = _run_parent_guarded
    print(
        "PHASE2A_RUNTIME_GUARD zero_root_fit=ACTIVE authoritative_policy_audit=ACTIVE "
        "guarded_children=ACTIVE parallel_policy_arms=3x2threads_per_seed",
        flush=True,
    )
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
