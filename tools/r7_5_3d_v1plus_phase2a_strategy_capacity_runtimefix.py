from __future__ import annotations

"""Mechanical execution guard for the frozen R7.5.3D Phase 2A ablation.

The Phase2A base runner deliberately reuses the admitted x4 pattern: collect four
64-root chunks, then execute exactly one frozen Advantage fit with a temporary
zero-root config.  The shared stage function still divides temporary reporting
fields by zero, so use the already-audited x4 fit-only helper that preserves the
same Advantage fit/audit/state semantics and emits zero placeholders until the
Phase2A runner patches them with the true 256-root totals.

This wrapper also pins the local AveragePolicy audit to the authoritative Phase-2
seed `training_seed ^ 0x71A5BEEF` for every capacity arm and makes the final
stage-report marker recoverable if power is lost after the atomic resume
checkpoint but before the small JSON report is written.

No scientific dimension, training seed, chance schedule, model, threshold,
reservoir capacity arm, or learner budget is changed here.
"""

import json
from pathlib import Path
import random

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
                # The resume checkpoint is the authoritative atomic state.  The
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


def _fit_seed_policies_authoritative_audit(*, seed_root: Path, training_seed: int, bundle, arms):
    policy_root = seed_root / "policies"
    policy_root.mkdir(parents=True, exist_ok=True)
    native_state = bundle.batch_rng.getstate()
    rows = {}
    for mode in ("COMMON_LEARNER", "NATIVE_LEARNER"):
        for arm_name in base.CAPACITIES:
            key = f"{mode}__{arm_name}"
            artifact = policy_root / f"{key}.pt"
            meta = policy_root / f"{key}.json"
            if artifact.is_file() and meta.is_file():
                saved = json.loads(meta.read_text(encoding="utf-8"))
                if (
                    saved.get("status") == "POLICY_FIT_COMPLETE"
                    and int(saved.get("training_seed", -1)) == int(training_seed)
                    and saved.get("authoritative_policy_audit_seed") == (int(training_seed) ^ 0x71A5BEEF)
                ):
                    rows[key] = saved
                    print(f"[Phase2A policy resume] seed={training_seed} {key}", flush=True)
                    continue
            if mode == "COMMON_LEARNER":
                init_seed = base.COMMON_POLICY_INIT_SEED
                rng = random.Random(base.COMMON_BATCH_SEED)
            else:
                init_seed = (int(training_seed) ^ 0x5DEECE66D) & 0x7FFFFFFF
                rng = random.Random()
                rng.setstate(native_state)
            audit_seed = int(training_seed) ^ 0x71A5BEEF
            print(f"[Phase2A policy fit] seed={training_seed} {key}", flush=True)
            model, fit = base._fit_policy(
                arms[arm_name],
                init_seed=init_seed,
                rng=rng,
                audit_seed=audit_seed,
            )
            payload = {
                "schema": base.SEED_SCHEMA,
                "status": "POLICY_FIT_COMPLETE",
                "representation": base.REPRESENTATION,
                "domain": base.DOMAIN,
                "training_seed": int(training_seed),
                "learner_mode": mode,
                "arm": arm_name,
                "capacity": base.CAPACITIES[arm_name],
                "authoritative_policy_audit_seed": int(audit_seed),
                "model_state": model.state_dict(),
                "fit": fit,
            }
            base._atomic_torch_save(payload, artifact)
            saved = {
                "schema": base.SEED_SCHEMA,
                "status": "POLICY_FIT_COMPLETE",
                "training_seed": int(training_seed),
                "learner_mode": mode,
                "arm": arm_name,
                "capacity": base.CAPACITIES[arm_name],
                "authoritative_policy_audit_seed": int(audit_seed),
                "artifact": str(artifact),
                "fit": fit,
            }
            base._atomic_json(saved, meta)
            rows[key] = saved
    return rows


def main() -> int:
    base.run_one_phase2_v3_iteration = _fit_only_iteration
    base._validate_stream_prefix = _validate_stream_prefix_recoverable
    base._fit_seed_policies = _fit_seed_policies_authoritative_audit
    print("PHASE2A_RUNTIME_GUARD zero_root_fit=ACTIVE authoritative_policy_audit=ACTIVE", flush=True)
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
