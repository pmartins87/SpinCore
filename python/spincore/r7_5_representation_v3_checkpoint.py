from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path

import torch

from spincore.r7_5_action_contract import postflop_candidate_specs
from spincore.r7_5_representation_v3 import (
    H2_FINAL,
    H3_FINAL,
    V3_REPRESENTATIONS,
    RepresentationV3Bundle,
    make_representation_v3_bundle,
)
from spincore_nn.reservoir import UniformReservoir

SCHEMA = "SPINCORE_R7_5_3C_REPRESENTATION_V3_CHECKPOINT_V1"


@dataclass
class RepresentationV3Progress:
    iteration: int = 0
    global_root: int = 0
    advantage_optimizer_step: int = 0
    policy_optimizer_step: int = 0
    phase: str = "collect"


def checkpoint_payload(
    bundle: RepresentationV3Bundle,
    progress: RepresentationV3Progress,
    *,
    domain: str,
    action_candidate: str,
    execution_sha: str,
    architecture_fingerprint_sha256: str,
    extra: dict | None = None,
) -> dict:
    if bundle.representation not in V3_REPRESENTATIONS:
        raise ValueError("only final H2/H3 V3 representations are checkpointable here")
    if not execution_sha.strip() or not architecture_fingerprint_sha256.strip():
        raise ValueError("checkpoint requires immutable execution/model identity")
    return {
        "schema": SCHEMA,
        "representation": bundle.representation,
        "domain": str(domain),
        "seed": int(bundle.seed),
        "action_candidate": str(action_candidate),
        "execution_sha": str(execution_sha),
        "architecture_fingerprint_sha256": str(architecture_fingerprint_sha256),
        "config": bundle.config.to_dict(),
        "advantage": bundle.advantage.state_dict(),
        "policy": bundle.policy.state_dict(),
        "adv_opt": bundle.adv_opt.state_dict(),
        "pol_opt": bundle.pol_opt.state_dict(),
        "adv_mem": bundle.adv_mem.state_dict(),
        "pol_mem": bundle.pol_mem.state_dict(),
        "batch_rng": bundle.batch_rng.getstate(),
        # Audit-only: V3 model/training correctness is intentionally independent
        # of the caller's global torch RNG. Loader must never restore this value.
        "global_torch_rng_audit": torch.get_rng_state().clone().cpu(),
        "counters": dict(bundle.counters),
        "progress": asdict(progress),
        "extra": dict(extra or {}),
        "production_training_authorized": False,
        "ready_for_tables": False,
    }


def save_representation_v3_checkpoint(
    path: str | Path,
    bundle: RepresentationV3Bundle,
    progress: RepresentationV3Progress,
    *,
    domain: str,
    action_candidate: str,
    execution_sha: str,
    architecture_fingerprint_sha256: str,
    extra: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(
        bundle,
        progress,
        domain=domain,
        action_candidate=action_candidate,
        execution_sha=execution_sha,
        architecture_fingerprint_sha256=architecture_fingerprint_sha256,
        extra=extra,
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_representation_v3_checkpoint(
    path: str | Path,
    *,
    repo_root: str | Path,
    expected_domain: str,
    expected_representation: str,
    expected_seed: int,
    expected_action_candidate: str,
    expected_execution_sha: str,
    expected_architecture_fingerprint_sha256: str,
    device: str = "cpu",
) -> tuple[RepresentationV3Bundle, RepresentationV3Progress, object, dict]:
    global_rng_before = torch.get_rng_state().clone()
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema") != SCHEMA:
        raise ValueError("wrong R7.5.3C V3 checkpoint schema")
    if bool(payload.get("ready_for_tables")) or bool(payload.get("production_training_authorized")):
        raise ValueError("V3 checkpoint illegally authorizes production/table use")

    expected = {
        "domain": str(expected_domain),
        "representation": str(expected_representation),
        "seed": int(expected_seed),
        "action_candidate": str(expected_action_candidate),
        "execution_sha": str(expected_execution_sha),
        "architecture_fingerprint_sha256": str(expected_architecture_fingerprint_sha256),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"V3 checkpoint identity mismatch for {key}: "
                f"{payload.get(key)!r} != {value!r}"
            )
    if str(expected_representation) not in (H2_FINAL, H3_FINAL):
        raise ValueError("unsupported final V3 representation")

    specs = postflop_candidate_specs(Path(repo_root))
    if expected_action_candidate not in specs:
        raise ValueError("unknown V3 checkpoint action candidate")
    action_spec = specs[expected_action_candidate]

    bundle = make_representation_v3_bundle(
        str(expected_representation),
        int(expected_seed),
        device=device,
        reservoir_capacity=int(payload["adv_mem"]["capacity"]),
        lr=float(payload["adv_opt"]["param_groups"][0]["lr"]),
    )
    if bundle.config.to_dict() != dict(payload["config"]):
        raise ValueError("V3 checkpoint network config mismatch")
    bundle.advantage.load_state_dict(payload["advantage"])
    bundle.policy.load_state_dict(payload["policy"])
    bundle.adv_opt.load_state_dict(payload["adv_opt"])
    bundle.pol_opt.load_state_dict(payload["pol_opt"])
    bundle.adv_mem = UniformReservoir.from_state_dict(payload["adv_mem"])
    bundle.pol_mem = UniformReservoir.from_state_dict(payload["pol_mem"])
    bundle.batch_rng.setstate(payload["batch_rng"])
    bundle.counters = dict(payload["counters"])

    # Loading must not modify global torch RNG. Model factories are fork_rng-
    # isolated and all continuation randomness lives in explicit Python RNGs.
    if not torch.equal(global_rng_before, torch.get_rng_state()):
        raise RuntimeError("loading V3 checkpoint changed global torch RNG")

    progress = RepresentationV3Progress(**dict(payload["progress"]))
    return bundle, progress, action_spec, dict(payload.get("extra") or {})
