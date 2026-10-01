#!/usr/bin/env python3
from __future__ import annotations

"""Predict every preregistered rebuild member on the frozen stability bank.

Each member applies its own fullpool/specialist route first. Only then are the
member probability tensors handed to the rebuild-stability evaluator. This tool
does not weight, drop, rank or select members.

PROJECT_CONTRACT_IDS:
MODEL-025,RNG-013,VALID-025,VALID-031,VALID-035,VALID-036,VALID-037,SAFE-011
"""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import audit_3h_fresh_semantic_strategy_distill_10105 as distill
from audit_3h_average_policy_semantic_continuation_10105 import (
    V1SemanticPolicyNet,
    semantic_vector_from_obs,
)
from spincore_nn.lean_batch import vectorized_batch

PROTOCOL = ROOT / "contracts" / "post15315_multirebuild_preregistration.json"
FULLPOOL_SCHEMA = "SPINCORE_POST15315_REBUILD_FULLPOOL_MEMBER_V1"
SPECIALIST_SCHEMA = "SPINCORE_POST15315_REBUILD_SPECIALIST_MEMBER_V1"
BANK_SCHEMA = "SPINCORE_POST15315_REBUILD_STABILITY_BANK_V1"
PRED_SCHEMA = "SPINCORE_POST15315_REBUILD_STABILITY_PREDICTIONS_V1"
FINAL_ITERATION = 15315


def load_protocol(path: Path) -> tuple[dict, str]:
    p = json.loads(path.read_text(encoding="utf-8"))
    if p.get("schema") != "SPINCORE_POST15315_MULTIREBUILD_PREREG_V1":
        raise RuntimeError("wrong post-15315 preregistration schema")
    return p, distill.sha256(path.resolve(strict=True))


def decode_metadata(z) -> dict:
    if "metadata_json" not in z.files:
        raise RuntimeError("stability bank missing metadata_json")
    raw = np.asarray(z["metadata_json"], dtype=np.uint8).tobytes()
    return json.loads(raw.decode("utf-8"))


def validate_bank(z, protocol_sha: str, protocol: dict) -> dict:
    meta = decode_metadata(z)
    if meta.get("schema") != BANK_SCHEMA:
        raise RuntimeError("wrong stability-bank schema")
    if meta.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("stability-bank protocol SHA mismatch")
    spec = protocol["shared_fresh_evaluation"]["stability_state_bank"]
    if int(meta.get("seed", -1)) != int(spec["seed"]):
        raise RuntimeError("stability-bank seed mismatch")
    if int(meta.get("forced_three_handed_episodes", -1)) != int(
        spec["forced_three_handed_episodes"]
    ):
        raise RuntimeError("stability-bank episode count mismatch")
    if bool(meta.get("training_use_permitted", True)):
        raise RuntimeError("stability bank unexpectedly permits training use")
    if bool(meta.get("member_selection_permitted", True)):
        raise RuntimeError("stability bank unexpectedly permits member selection")
    return meta


def load_member(
    member_dir: Path,
    member: int,
    *,
    protocol: dict,
    protocol_sha: str,
    teacher_sha: str,
):
    full_path = member_dir / f"member_{member:02d}_fullpool.pt"
    spec_path = member_dir / f"member_{member:02d}_specialist.pt"
    fp = torch.load(full_path.resolve(strict=True), map_location="cpu", weights_only=False)
    sp = torch.load(spec_path.resolve(strict=True), map_location="cpu", weights_only=False)

    expected_seeds = next(
        dict(x)
        for x in protocol["rebuild_plan"]["member_seeds"]
        if int(x["member"]) == int(member)
    )
    for payload, schema, label in (
        (fp, FULLPOOL_SCHEMA, "fullpool"),
        (sp, SPECIALIST_SCHEMA, "specialist"),
    ):
        if payload.get("schema") != schema:
            raise RuntimeError(f"member {member} wrong {label} schema")
        if int(payload.get("member", -1)) != int(member):
            raise RuntimeError(f"member {member} {label} member-id mismatch")
        if int(payload.get("semantic_completed_iteration", -1)) != FINAL_ITERATION:
            raise RuntimeError(f"member {member} {label} iteration mismatch")
        if payload.get("teacher_sha256") != teacher_sha:
            raise RuntimeError(f"member {member} {label} teacher mismatch")
        if payload.get("protocol_sha256") != protocol_sha:
            raise RuntimeError(f"member {member} {label} protocol mismatch")
        if dict(payload.get("member_seeds") or {}) != expected_seeds:
            raise RuntimeError(f"member {member} {label} seed mismatch")

    avg = protocol["rebuild_plan"]["average_policy"]
    spec_cfg = protocol["rebuild_plan"]["specialist"]
    if int(fp.get("selected_steps", -1)) != int(avg["fullpool_fit_steps"]):
        raise RuntimeError(f"member {member} fullpool step-budget mismatch")
    if int(sp.get("selected_steps", -1)) != int(spec_cfg["fit_steps"]):
        raise RuntimeError(f"member {member} specialist step-budget mismatch")
    if sp.get("route_contract") != spec_cfg["route"]:
        raise RuntimeError(f"member {member} specialist route mismatch")
    if sp.get("base_fullpool_sha256") != distill.sha256(full_path):
        raise RuntimeError(f"member {member} specialist/fullpool hash mismatch")

    full = V1SemanticPolicyNet()
    full.load_state_dict(fp["model_state"])
    full.eval()
    specialist = V1SemanticPolicyNet()
    specialist.load_state_dict(sp["model_state"])
    specialist.eval()
    return full, specialist, full_path, spec_path


def chunk_samples(observations, legal, targets, start: int, end: int):
    out = []
    for i in range(start, end):
        out.append(
            distill.ActionStrategySample(
                observation=np.asarray(observations[i], dtype=np.uint8).tobytes(),
                legal=tuple(bool(x) for x in legal[i]),
                target=tuple(float(x) for x in targets[i]),
                weight=1.0,
                iteration=FINAL_ITERATION,
            )
        )
    return out


def routed_probs(
    fullpool,
    specialist,
    observations,
    legal,
    targets,
    route_mask,
    *,
    batch_size: int,
) -> np.ndarray:
    parts = []
    n = int(observations.shape[0])
    for start in range(0, n, int(batch_size)):
        end = min(n, start + int(batch_size))
        samples = chunk_samples(observations, legal, targets, start, end)
        b, _t, _w = vectorized_batch(samples, "cpu")
        sb = dict(b)
        sb["semantic"] = torch.tensor(
            np.asarray(
                [semantic_vector_from_obs(s.observation) for s in samples],
                dtype=np.float32,
            ),
            dtype=torch.float32,
        )
        with torch.no_grad():
            p = fullpool.probabilities(sb).detach().cpu().numpy()
            local_route = np.asarray(route_mask[start:end], dtype=bool)
            if np.any(local_route):
                ps = specialist.probabilities(sb).detach().cpu().numpy()
                p[local_route] = ps[local_route]
        parts.append(np.asarray(p, dtype=np.float32))
    out = np.concatenate(parts, axis=0)
    if out.shape != (n, 10):
        raise RuntimeError(f"prediction shape drift: {out.shape}")
    return out


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    if tmp.exists():
        tmp.unlink()
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", type=Path, default=PROTOCOL)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--member-dir", type=Path, required=True)
    ap.add_argument("--members", type=int, choices=(8, 12), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4096)
    args = ap.parse_args()

    protocol, protocol_sha = load_protocol(args.protocol.resolve(strict=True))
    allowed = int(protocol["rebuild_plan"]["primary_rebuild_count"])
    maximum = int(protocol["rebuild_plan"]["maximum_rebuild_count"])
    if int(args.members) not in (allowed, maximum):
        raise RuntimeError("only preregistered K=8 or K=12 is permitted")

    torch.set_num_threads(int(args.threads))
    bank_sha = distill.sha256(args.bank.resolve(strict=True))
    with np.load(args.bank.resolve(strict=True), allow_pickle=False) as z:
        meta = validate_bank(z, protocol_sha, protocol)
        observations = np.asarray(z["observations"], dtype=np.uint8)
        legal = np.asarray(z["legal"], dtype=bool)
        targets = np.asarray(z["targets"], dtype=np.float32)
        s72 = np.asarray(z["surface_72o"], dtype=bool)
        trips = np.asarray(z["surface_trips_fold"], dtype=bool)
        hc = np.asarray(z["surface_high_card_no_draw"], dtype=bool)

    n = int(observations.shape[0])
    if legal.shape != (n, 10) or targets.shape != (n, 10):
        raise RuntimeError("stability bank native-carrier shape mismatch")
    if np.any(legal[:, [2, 4, 6]]):
        raise RuntimeError("stability bank exposes dormant action slot")

    teacher_sha = str(meta["teacher_sha256"])
    member_probs = []
    member_meta = []
    for member in range(1, int(args.members) + 1):
        full, specialist, full_path, spec_path = load_member(
            args.member_dir.resolve(),
            member,
            protocol=protocol,
            protocol_sha=protocol_sha,
            teacher_sha=teacher_sha,
        )
        p = routed_probs(
            full,
            specialist,
            observations,
            legal,
            targets,
            trips,
            batch_size=int(args.batch_size),
        )
        illegal = np.where(legal, 0.0, p)
        if float(np.max(illegal)) > 1e-7:
            raise RuntimeError(f"member {member} puts mass on illegal action")
        if not np.allclose(p.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
            raise RuntimeError(f"member {member} probability rows do not sum to one")
        member_probs.append(p)
        member_meta.append(
            {
                "member": member,
                "fullpool_sha256": distill.sha256(full_path),
                "specialist_sha256": distill.sha256(spec_path),
            }
        )
        print(
            f"POST15315_STABILITY_PREDICT member={member}/{args.members}",
            flush=True,
        )

    probs = np.stack(member_probs, axis=0).astype(np.float32)
    meta_out = {
        "schema": PRED_SCHEMA,
        "protocol_sha256": protocol_sha,
        "bank_sha256": bank_sha,
        "teacher_sha256": teacher_sha,
        "member_count": int(args.members),
        "members": member_meta,
        "route": protocol["rebuild_plan"]["specialist"]["route"],
        "aggregation_preselection_performed": False,
        "member_weighting_performed": False,
        "member_dropping_performed": False,
        "action_carrier": protocol["aggregation"]["action_carrier"],
    }
    atomic_npz(
        args.out,
        probs=probs,
        legal=legal,
        surface_72o=s72,
        surface_trips_fold=trips,
        surface_high_card_no_draw=hc,
        member_ids=np.arange(1, int(args.members) + 1, dtype=np.int16),
        metadata_json=np.frombuffer(
            json.dumps(meta_out, sort_keys=True).encode("utf-8"),
            dtype=np.uint8,
        ),
    )
    print("POST15315_REBUILD_STABILITY_PREDICTIONS_PASS")
    print(f"out={args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
