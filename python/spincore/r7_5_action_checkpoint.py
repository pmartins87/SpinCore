from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path

import torch

from spincore.r7_5_action_contract import (
    ActionCandidateSpec,
    postflop_candidate_specs,
    preflop_candidate_specs,
)
from spincore.r7_5_action_training import ActionDomainBundle, make_action_bundle
from spincore_nn.reservoir import UniformReservoir

SCHEMA = "SPINCORE_R7_5_ACTION_CHECKPOINT_V1"


@dataclass
class ActionProgress:
    iteration: int = 0
    phase: str = "collect"
    root_index: int = 0
    traverser_index: int = 0
    advantage_optimizer_step: int = 0
    policy_optimizer_step: int = 0


def _candidate_spec(
    repo_root: str | Path,
    *,
    phase: str,
    candidate_id: str,
    selected_postflop_candidate: str | None,
) -> ActionCandidateSpec:
    root = Path(repo_root)
    if phase == "R7_5_4A_POSTFLOP":
        specs = postflop_candidate_specs(root)
    elif phase == "R7_5_4B_PREFLOP":
        if not selected_postflop_candidate:
            raise ValueError("preflop checkpoint requires selected R7.5.4A candidate")
        specs = preflop_candidate_specs(
            root,
            selected_postflop_candidate=selected_postflop_candidate,
        )
    else:
        raise ValueError(f"unsupported R7.5.4 checkpoint phase: {phase!r}")
    try:
        return specs[candidate_id]
    except KeyError as exc:
        raise ValueError(f"unknown R7.5.4 action candidate: {candidate_id!r}") from exc


def checkpoint_payload(
    bundle: ActionDomainBundle,
    progress: ActionProgress,
    *,
    action_phase: str,
    selected_postflop_candidate: str | None = None,
    extra: dict | None = None,
) -> dict:
    return {
        "schema": SCHEMA,
        "domain": bundle.domain,
        "seed": int(bundle.seed),
        "selected_representation": bundle.selected_representation,
        "action_candidate": bundle.action_candidate,
        "action_phase": str(action_phase),
        "selected_postflop_candidate": selected_postflop_candidate,
        "config": bundle.config.to_dict(),
        "advantage": bundle.advantage.state_dict(),
        "policy": bundle.policy.state_dict(),
        "adv_opt": bundle.adv_opt.state_dict(),
        "pol_opt": bundle.pol_opt.state_dict(),
        "adv_mem": bundle.adv_mem.state_dict(),
        "pol_mem": bundle.pol_mem.state_dict(),
        "batch_rng": bundle.batch_rng.getstate(),
        "torch_rng": torch.get_rng_state().clone().cpu(),
        "counters": dict(bundle.counters),
        "progress": asdict(progress),
        "extra": dict(extra or {}),
        "ready_for_tables": False,
    }


def save_action_checkpoint(
    path: str | Path,
    bundle: ActionDomainBundle,
    progress: ActionProgress,
    *,
    action_phase: str,
    selected_postflop_candidate: str | None = None,
    extra: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(
        bundle,
        progress,
        action_phase=action_phase,
        selected_postflop_candidate=selected_postflop_candidate,
        extra=extra,
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_action_checkpoint(
    path: str | Path,
    *,
    repo_root: str | Path,
    device: str = "cpu",
) -> tuple[ActionDomainBundle, ActionProgress, ActionCandidateSpec, dict]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema") != SCHEMA:
        raise ValueError("wrong R7.5 action checkpoint schema")
    if bool(payload.get("ready_for_tables")):
        raise ValueError("R7.5 action checkpoint illegally authorizes table use")

    action_spec = _candidate_spec(
        repo_root,
        phase=str(payload["action_phase"]),
        candidate_id=str(payload["action_candidate"]),
        selected_postflop_candidate=payload.get("selected_postflop_candidate"),
    )
    bundle = make_action_bundle(
        int(payload["seed"]),
        domain=str(payload["domain"]),
        selected_representation=str(payload["selected_representation"]),
        action_spec=action_spec,
        device=device,
        reservoir_capacity=int(payload["adv_mem"]["capacity"]),
        lr=float(payload["adv_opt"]["param_groups"][0]["lr"]),
    )
    if bundle.config.to_dict() != dict(payload["config"]):
        raise ValueError("R7.5 action checkpoint config mismatch")

    bundle.advantage.load_state_dict(payload["advantage"])
    bundle.policy.load_state_dict(payload["policy"])
    bundle.adv_opt.load_state_dict(payload["adv_opt"])
    bundle.pol_opt.load_state_dict(payload["pol_opt"])
    bundle.adv_mem = UniformReservoir.from_state_dict(payload["adv_mem"])
    bundle.pol_mem = UniformReservoir.from_state_dict(payload["pol_mem"])
    bundle.batch_rng.setstate(payload["batch_rng"])
    bundle.counters = dict(payload["counters"])
    torch.set_rng_state(payload["torch_rng"].cpu())

    progress = ActionProgress(**dict(payload["progress"]))
    return bundle, progress, action_spec, dict(payload.get("extra") or {})
