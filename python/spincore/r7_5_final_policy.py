from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from spincore.r7_5_action_checkpoint import load_action_checkpoint
from spincore.r7_5_action_cfr import legal_mask
from spincore.r7_5_action_stage import FINAL_REPORT_SCHEMA
from spincore.r7_5_action_stage_contract import (
    ITERATIONS,
    POLICY_STEPS,
    RESERVOIR_CAPACITY,
    SELECTED_REPRESENTATION,
)
from spincore_nn.action_models import collate_action_observations


@dataclass
class FinalizedActionPolicy:
    bundle: object
    action_spec: object
    final_report: dict
    execution_sha: str
    checkpoint_path: str

    @property
    def candidate_id(self) -> str:
        return str(self.bundle.action_candidate)

    @property
    def domain(self) -> str:
        return str(self.bundle.domain)

    @property
    def training_seed(self) -> int:
        return int(self.bundle.seed)

    def __call__(self, _state, observation: bytes, legal: tuple[int, ...]) -> tuple[float, ...]:
        mask = legal_mask(legal)
        batch = collate_action_observations(
            SELECTED_REPRESENTATION,
            [observation],
            [mask],
            device="cpu",
        )
        self.bundle.policy.eval()
        with torch.no_grad():
            logits = self.bundle.policy(batch).masked_fill(~batch["legal"], -1e9)
            probabilities = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()
        out = tuple(float(value) for value in probabilities)
        if len(out) != 10:
            raise RuntimeError("final action policy emitted non-ten-action distribution")
        return out


def load_finalized_action_policy(
    checkpoint_path: str | Path,
    *,
    repo_root: str | Path,
    expected_execution_sha: str,
    expected_candidate_id: str | None = None,
    expected_domain: str | None = None,
    expected_training_seed: int | None = None,
) -> FinalizedActionPolicy:
    """Load one final R7.5.4 policy without perturbing caller Torch RNG.

    load_action_checkpoint restores the checkpoint Torch RNG as required for
    training resume. Evaluation must be observational, so this wrapper saves and
    restores the caller's global Torch RNG around the load operation.
    """
    if not str(expected_execution_sha).strip():
        raise ValueError("expected immutable execution SHA is required")
    torch_rng = torch.get_rng_state().clone()
    try:
        bundle, progress, action_spec, extra = load_action_checkpoint(
            checkpoint_path,
            repo_root=repo_root,
            device="cpu",
        )
    finally:
        torch.set_rng_state(torch_rng)

    if progress.phase != "post_policy_fit" or int(progress.iteration) != ITERATIONS:
        raise ValueError("checkpoint is not a finalized iteration-5 action policy")
    if int(progress.policy_optimizer_step) != POLICY_STEPS:
        raise ValueError("final AveragePolicy optimizer-step count mismatch")
    if str(bundle.selected_representation) != SELECTED_REPRESENTATION:
        raise ValueError("final policy is not bound to durable C0 representation")
    if int(bundle.pol_mem.capacity) != RESERVOIR_CAPACITY or int(bundle.adv_mem.capacity) != RESERVOIR_CAPACITY:
        raise ValueError("final action reservoir capacity mismatch")
    if str(extra.get("execution_sha", "")) != str(expected_execution_sha):
        raise ValueError("final action checkpoint execution SHA mismatch")

    final = dict(extra.get("final_report") or {})
    if final.get("schema") != FINAL_REPORT_SCHEMA:
        raise ValueError("final action checkpoint is missing the durable final report")
    if final.get("candidate_id") != bundle.action_candidate:
        raise ValueError("final report candidate identity mismatch")
    if final.get("domain") != bundle.domain:
        raise ValueError("final report domain identity mismatch")
    if int(final.get("training_seed", -1)) != int(bundle.seed):
        raise ValueError("final report training-seed identity mismatch")
    if final.get("selected_representation") != SELECTED_REPRESENTATION:
        raise ValueError("final report representation mismatch")
    if int(final.get("roots", -1)) != 160:
        raise ValueError("finalized policy is not the R7.5.4A-160 artifact")
    if int(final.get("average_policy_optimizer_steps", -1)) != POLICY_STEPS:
        raise ValueError("final report AveragePolicy step count mismatch")
    if bool(final.get("strategic_selection_permitted_at_160")):
        raise ValueError("160-root artifact illegally permits final strategic selection")
    if bool(final.get("production_training_authorized")) or bool(final.get("ready_for_tables")):
        raise ValueError("R7.5.4A checkpoint illegally authorizes production/table use")

    if expected_candidate_id is not None and str(bundle.action_candidate) != str(expected_candidate_id):
        raise ValueError("final policy candidate differs from expected identity")
    if expected_domain is not None and str(bundle.domain) != str(expected_domain):
        raise ValueError("final policy domain differs from expected identity")
    if expected_training_seed is not None and int(bundle.seed) != int(expected_training_seed):
        raise ValueError("final policy training seed differs from expected identity")

    return FinalizedActionPolicy(
        bundle=bundle,
        action_spec=action_spec,
        final_report=final,
        execution_sha=str(expected_execution_sha),
        checkpoint_path=str(Path(checkpoint_path)),
    )
