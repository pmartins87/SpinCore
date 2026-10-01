#!/usr/bin/env python3
from __future__ import annotations

"""Build one frozen post-15315 AveragePolicy/fullpool/specialist rebuild member.

This is a construction tool only. It performs no strength benchmark and no
result-dependent member selection. Member identity, all collection/fit seeds,
budgets, route, and support minima come from the preregistration frozen before
teacher 15315 is observed.

PROJECT_CONTRACT_IDS:
TRAIN-022,TRAIN-023,MODEL-025,RNG-001,RNG-003,RNG-013,
VALID-025,VALID-031,VALID-032,VALID-034,VALID-035,VALID-036,VALID-037,VALID-038,
SAFE-001,SAFE-010,SAFE-011,PERF-001,PERF-002,CKPT-001,ART-001
"""

import argparse
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import audit_3h_fresh_semantic_strategy_distill_10105 as distill
import audit_3h_fresh_semantic_strategy_independent500_10105 as confirm
import audit_3h_semantic_stratified_specialist_moe_10115 as spec
import audit_3h_semantic_tail_strong_hand_coverage_10115 as coverage
import post15315_collection as collect
from audit_3h_average_policy_semantic_continuation_10105 import (
    semantic_vector_from_obs,
)
from spincore.solver import SolverLibrary
from spincore_nn.lean_batch import vectorized_batch
from spincore_nn.training import train_step

PROTOCOL = ROOT / "contracts" / "post15315_multirebuild_preregistration.json"
EXPECTED_CHECKPOINT_SHA = distill.EXPECTED_SHA
ADV_SCHEMA = "SPINCORE_3H_SEMANTIC_LONG_ENSEMBLE_V1"
TAIL_SCHEMA = "SPINCORE_POST15315_REBUILD_TAIL_MEMBER_V1"
FULLPOOL_SCHEMA = "SPINCORE_POST15315_REBUILD_FULLPOOL_MEMBER_V1"
SPECIALIST_SCHEMA = "SPINCORE_POST15315_REBUILD_SPECIALIST_MEMBER_V1"
REPORT_SCHEMA = "SPINCORE_POST15315_REBUILD_MEMBER_REPORT_V1"
DOMAIN = "THREE_HANDED"
FINAL_ITERATION = 15315


def load_protocol(path: Path) -> dict:
    p = json.loads(path.read_text(encoding="utf-8"))
    if p.get("schema") != "SPINCORE_POST15315_MULTIREBUILD_PREREG_V1":
        raise RuntimeError("wrong post-15315 preregistration schema")
    if int(p["teacher"]["required_iteration"]) != FINAL_ITERATION:
        raise RuntimeError("post-15315 protocol target drift")
    return p


def member_spec(protocol: dict, member: int) -> dict:
    rows = list(protocol["rebuild_plan"]["member_seeds"])
    for row in rows:
        if int(row["member"]) == int(member):
            return dict(row)
    raise RuntimeError(f"member {member} not preregistered")


def load_teacher(path: Path, expected_sha256: str):
    actual = distill.sha256(path.resolve(strict=True))
    if actual != str(expected_sha256):
        raise RuntimeError(
            "semantic Advantage teacher SHA mismatch: "
            f"expected={expected_sha256} actual={actual}"
        )
    p = torch.load(path.resolve(strict=True), map_location="cpu", weights_only=False)
    if p.get("schema") != ADV_SCHEMA:
        raise RuntimeError(f"wrong semantic Advantage schema: {p.get('schema')!r}")
    if p.get("source_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("semantic Advantage source checkpoint mismatch")
    if int(p.get("completed_iteration", -1)) != FINAL_ITERATION:
        raise RuntimeError(
            f"teacher iteration mismatch: {p.get('completed_iteration')!r}"
        )
    states = list(p.get("members") or [])
    if len(states) != 8:
        raise RuntimeError(f"expected eight teacher members, got {len(states)}")
    return [distill.load_semantic_advantage(s) for s in states], p, actual


def strong(sample) -> bool:
    m = coverage.sample_meta(sample)
    return bool(
        int(m["street"]) > 0
        and int(m["made"]) >= 3
        and bool(m["fold_legal"])
    )


def skey(sample):
    return bytes(sample.observation), tuple(bool(x) for x in sample.legal)


def train_semantic(
    model,
    optimizer,
    samples,
    cfg,
    *,
    steps: int,
    seed: int,
    label: str,
) -> None:
    if not samples:
        raise RuntimeError(f"{label}: empty training sample")
    rng = random.Random(int(seed))
    bs = int(cfg["batch_size"])
    for step in range(int(steps)):
        idx = rng.sample(range(len(samples)), min(bs, len(samples)))
        ss = [samples[i] for i in idx]
        b, t, w = vectorized_batch(ss, "cpu")
        sb = dict(b)
        sb["semantic"] = torch.tensor(
            np.asarray(
                [semantic_vector_from_obs(s.observation) for s in ss],
                dtype=np.float32,
            ),
            dtype=torch.float32,
        )
        train_step(model, optimizer, sb, t, w, "strategy")
        if (step + 1) % 100 == 0 or step + 1 == int(steps):
            print(f"{label} {step + 1}/{steps}", flush=True)


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    torch.save(payload, tmp)
    os.replace(tmp, path)


def atomic_json_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--member", type=int, required=True)
    ap.add_argument("--protocol", type=Path, default=PROTOCOL)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--semantic-advantage", type=Path, required=True)
    ap.add_argument("--teacher-sha256", required=True)
    ap.add_argument("--solver", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--collection-threads", type=int, default=1)
    ap.add_argument("--fit-threads", type=int, default=8)
    ap.add_argument(
        "--rerun-identical-member",
        action="store_true",
        help="Allow replacement only for an explicitly identical-seed mechanical rerun.",
    )
    args = ap.parse_args()

    protocol = load_protocol(args.protocol.resolve(strict=True))
    max_k = int(protocol["rebuild_plan"]["maximum_rebuild_count"])
    if not 1 <= int(args.member) <= max_k:
        raise RuntimeError(f"member must be 1..{max_k}")
    seeds = member_spec(protocol, int(args.member))

    avg = protocol["rebuild_plan"]["average_policy"]
    sp = protocol["rebuild_plan"]["specialist"]
    base_episodes = int(avg["ordinary_teacher_episodes_per_rebuild"])
    base_steps = int(avg["ordinary_fit_steps"])
    augment_episodes = int(avg["strong_augmentation_teacher_episodes_per_rebuild"])
    fullpool_steps = int(avg["fullpool_fit_steps"])
    specialist_episodes = int(sp["teacher_episodes_per_rebuild"])
    specialist_steps = int(sp["fit_steps"])

    cp = args.checkpoint.resolve(strict=True)
    actual_cp = distill.sha256(cp)
    if actual_cp != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError(
            f"source 10105 checkpoint SHA mismatch: {actual_cp}"
        )
    payload = torch.load(cp, map_location="cpu", weights_only=False)
    d3 = (payload.get("domains") or {}).get(DOMAIN) or {}
    cfg = dict(payload.get("config") or {})
    if not d3 or not cfg:
        raise RuntimeError("frozen 10105 THREE_HANDED state/config missing")

    torch.set_num_threads(int(args.collection_threads))
    teacher, teacher_payload, teacher_sha = load_teacher(
        args.semantic_advantage, args.teacher_sha256
    )
    solver = SolverLibrary(args.solver.resolve(strict=True))

    out = args.out_dir.resolve()
    tail_path = out / f"member_{args.member:02d}_tail.pt"
    fullpool_path = out / f"member_{args.member:02d}_fullpool.pt"
    specialist_path = out / f"member_{args.member:02d}_specialist.pt"
    report_path = out / f"member_{args.member:02d}_report.json"
    outputs = [tail_path, fullpool_path, specialist_path, report_path]
    existing = [str(p) for p in outputs if p.exists()]
    if existing and not args.rerun_identical_member:
        raise RuntimeError(
            "refusing to overwrite durable member artifacts; "
            "use --rerun-identical-member only for a documented mechanical rerun: "
            + ", ".join(existing)
        )

    # Ordinary fresh teacher-target stream.
    base_train, base_holdout, base_collection_raw = collect.collect_fresh_split(
        solver,
        teacher,
        base_episodes,
        master_seed=int(seeds["base_collection_seed"]),
        sample_iteration=FINAL_ITERATION,
        progress_prefix=f"POST15315_MEMBER_{args.member:02d}_ORDINARY",
    )
    min_train = int(avg["min_ordinary_train_samples"])
    min_holdout = int(avg["min_ordinary_holdout_samples"])
    if len(base_train) < min_train or len(base_holdout) < min_holdout:
        raise RuntimeError(
            "ordinary target support below frozen construction minimum: "
            f"train={len(base_train)}/{min_train} "
            f"holdout={len(base_holdout)}/{min_holdout}"
        )

    _, _, tail, tailopt = confirm.initial_policy_pair(d3, cfg)
    torch.set_num_threads(int(args.fit_threads))
    train_semantic(
        tail,
        tailopt,
        base_train,
        cfg,
        steps=base_steps,
        seed=int(seeds["tail_fit_seed"]),
        label=f"POST15315_MEMBER_{args.member:02d}_TAIL_FIT",
    )
    tail.eval()

    # Independent strong-state augmentation for the fullpool general model.
    base_strong = [s for s in base_train if strong(s)]
    base_keys = {skey(s) for s in base_strong}
    torch.set_num_threads(int(args.collection_threads))
    aug_train, aug_hold, augment_collection_raw = collect.collect_strong_split(
        solver,
        teacher,
        augment_episodes,
        master_seed=int(seeds["augment_collection_seed"]),
        sample_iteration=FINAL_ITERATION,
        progress_prefix=f"POST15315_MEMBER_{args.member:02d}_AUGMENT",
    )
    seen = set()
    novel = []
    for s in list(aug_train) + list(aug_hold):
        key = skey(s)
        if key in base_keys or key in seen:
            continue
        seen.add(key)
        novel.append(s)
    min_novel = int(avg["min_novel_strong_states"])
    if len(novel) < min_novel:
        raise RuntimeError(
            "novel strong support below frozen construction minimum: "
            f"{len(novel)}/{min_novel}"
        )

    full_train = list(base_train) + novel
    _, _, fullpool, fullopt = confirm.initial_policy_pair(d3, cfg)
    torch.set_num_threads(int(args.fit_threads))
    train_semantic(
        fullpool,
        fullopt,
        full_train,
        cfg,
        steps=fullpool_steps,
        seed=int(seeds["fullpool_fit_seed"]),
        label=f"POST15315_MEMBER_{args.member:02d}_FULLPOOL_FIT",
    )
    fullpool.eval()

    # Frozen target-stratified specialist, no mode/step selection.
    torch.set_num_threads(int(args.collection_threads))
    sp_train, sp_hold, specialist_collection_raw = collect.collect_strong_split(
        solver,
        teacher,
        specialist_episodes,
        master_seed=int(seeds["specialist_collection_seed"]),
        sample_iteration=FINAL_ITERATION,
        progress_prefix=f"POST15315_MEMBER_{args.member:02d}_SPECIALIST",
    )
    unique = spec.unique_strong(list(sp_train) + list(sp_hold))
    strata = spec.stratum_counts(unique)
    if len(unique) < int(sp["min_unique_strong_states"]):
        raise RuntimeError(
            f"specialist unique support too small: {len(unique)}"
        )
    if int(strata.get("high", 0)) < int(sp["min_high_target_states"]):
        raise RuntimeError(f"specialist high-target support too small: {strata}")
    if int(strata.get("mid", 0)) < int(sp["min_mid_target_states"]):
        raise RuntimeError(f"specialist mid-target support too small: {strata}")

    torch.set_num_threads(int(args.fit_threads))
    specialist = spec.clone_model(fullpool)
    specialist_opt = torch.optim.Adam(
        specialist.parameters(), lr=float(cfg["learning_rate"])
    )
    specialist_rng = random.Random(int(seeds["specialist_fit_seed"]))
    spec.train_range(
        specialist,
        specialist_opt,
        unique,
        cfg,
        0,
        specialist_steps,
        specialist_rng,
        "STRATIFIED",
    )
    specialist.eval()

    common = {
        "source_checkpoint_sha256": actual_cp,
        "teacher_sha256": teacher_sha,
        "teacher_schema": teacher_payload.get("schema"),
        "semantic_completed_iteration": FINAL_ITERATION,
        "member": int(args.member),
        "member_seeds": seeds,
        "collection_threads": int(args.collection_threads),
        "fit_threads": int(args.fit_threads),
        "production_status": "BUILT_UNVALIDATED_NOT_PROMOTED",
    }

    atomic_torch_save(
        {
            "schema": TAIL_SCHEMA,
            **common,
            "selected_steps": base_steps,
            "ordinary_train_samples": len(base_train),
            "model_state": {k: v.detach().cpu() for k, v in tail.state_dict().items()},
        },
        tail_path,
    )
    atomic_torch_save(
        {
            "schema": FULLPOOL_SCHEMA,
            **common,
            "selected_steps": fullpool_steps,
            "ordinary_train_samples": len(base_train),
            "extra_unique_strong_states": len(novel),
            "model_state": {
                k: v.detach().cpu() for k, v in fullpool.state_dict().items()
            },
        },
        fullpool_path,
    )
    # Bind the specialist to the exact durable fullpool artifact on its first write.
    fullpool_sha = distill.sha256(fullpool_path)
    atomic_torch_save(
        {
            "schema": SPECIALIST_SCHEMA,
            **common,
            "selected_steps": specialist_steps,
            "selected_training_mode": "STRATIFIED",
            "specialist_train_episodes": specialist_episodes,
            "specialist_unique_strong_states": len(unique),
            "specialist_unique_strong_strata": strata,
            "base_fullpool_sha256": fullpool_sha,
            "route_contract": sp["route"],
            "model_state": {
                k: v.detach().cpu() for k, v in specialist.state_dict().items()
            },
        },
        specialist_path,
    )

    report = {
        "schema": REPORT_SCHEMA,
        "scope": "POST15315_PREREGISTERED_MEMBER_CONSTRUCTION_ONLY",
        **common,
        "ordinary_collection": collect.canonical_collection_stats(base_collection_raw),
        "ordinary_train_samples": len(base_train),
        "ordinary_holdout_samples": len(base_holdout),
        "ordinary_train_digest": collect.sample_digest(base_train),
        "ordinary_holdout_digest": collect.sample_digest(base_holdout),
        "base_strong_states": len(base_strong),
        "augmentation_collection": collect.canonical_collection_stats(
            augment_collection_raw
        ),
        "novel_strong_states": len(novel),
        "novel_strong_digest": collect.sample_digest(novel),
        "specialist_collection": collect.canonical_collection_stats(
            specialist_collection_raw
        ),
        "specialist_unique_strong_states": len(unique),
        "specialist_unique_strong_strata": strata,
        "specialist_unique_digest": collect.sample_digest(unique),
        "tail_artifact": str(tail_path),
        "tail_sha256": distill.sha256(tail_path),
        "fullpool_artifact": str(fullpool_path),
        "fullpool_sha256": fullpool_sha,
        "specialist_artifact": str(specialist_path),
        "specialist_sha256": distill.sha256(specialist_path),
        "construction_pass": True,
        "selection_performed": False,
        "strength_benchmark_performed": False,
        "interpretation": (
            "A construction PASS means this preregistered seed member was built "
            "with the frozen architecture and support minima. It does not admit "
            "the member or ensemble; every seeded member remains in the frozen "
            "set and the common stability/quality/safety gates decide the next step."
        ),
    }
    atomic_json_save(report, report_path)

    print(
        f"POST15315_REBUILD_MEMBER_{int(args.member):02d}_CONSTRUCTION_PASS",
        flush=True,
    )
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
