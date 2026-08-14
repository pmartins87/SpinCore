from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from spincore.r7_5_action_checkpoint import load_action_checkpoint
from spincore.r7_5_action_cfr import legal_mask, validate_policy
from spincore.r7_5_action_stage import FINAL_REPORT_SCHEMA
from spincore.r7_5_action_stage_contract import (
    ITERATIONS,
    POLICY_STEPS,
    RESERVOIR_CAPACITY,
    SELECTED_REPRESENTATION,
)
from spincore_nn.action_models import collate_action_observations

DEFAULT_EVALUATION_BATCH_SIZE = 256
DEFAULT_EXPECTED_ROOT_LEVEL = 160
ALLOWED_EVALUATION_ROOT_LEVELS = (160, 320, 640)


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

    def batch_probabilities(
        self,
        observations: Sequence[bytes],
        legal_sets: Sequence[tuple[int, ...]],
        *,
        batch_size: int = DEFAULT_EVALUATION_BATCH_SIZE,
    ) -> tuple[tuple[float, ...], ...]:
        if len(observations) != len(legal_sets):
            raise ValueError("final policy batch observation/legal count mismatch")
        if not observations:
            return ()
        width = int(batch_size)
        if width <= 0:
            raise ValueError("positive final-policy evaluation batch size required")
        out: list[tuple[float, ...]] = []
        self.bundle.policy.eval()
        for start in range(0, len(observations), width):
            obs_chunk = observations[start:start + width]
            legal_chunk = legal_sets[start:start + width]
            masks = [legal_mask(legal) for legal in legal_chunk]
            batch = collate_action_observations(
                SELECTED_REPRESENTATION,
                obs_chunk,
                masks,
                device="cpu",
            )
            with torch.no_grad():
                logits = self.bundle.policy(batch).masked_fill(~batch["legal"], -1e9)
                probabilities = torch.softmax(logits, dim=-1).detach().cpu().tolist()
            for raw, legal in zip(probabilities, legal_chunk):
                row = validate_policy(tuple(float(value) for value in raw), tuple(legal))
                out.append(row)
        if len(out) != len(observations):
            raise RuntimeError("final policy batch output count drift")
        return tuple(out)

    def __call__(self, _state, observation: bytes, legal: tuple[int, ...]) -> tuple[float, ...]:
        return self.batch_probabilities([observation], [legal], batch_size=1)[0]


def _validated_root_level(value: int) -> int:
    root_level = int(value)
    if root_level not in ALLOWED_EVALUATION_ROOT_LEVELS:
        raise ValueError(
            f"unsupported finalized-policy root level {root_level}; "
            f"expected one of {ALLOWED_EVALUATION_ROOT_LEVELS}"
        )
    return root_level


def load_finalized_action_policy(
    checkpoint_path: str | Path,
    *,
    repo_root: str | Path,
    expected_execution_sha: str,
    expected_root_level: int = DEFAULT_EXPECTED_ROOT_LEVEL,
    expected_candidate_id: str | None = None,
    expected_domain: str | None = None,
    expected_training_seed: int | None = None,
) -> FinalizedActionPolicy:
    """Load one final R7.5.4 policy without perturbing caller Torch RNG.

    The historical/default binding remains the 160-root pruning artifact. Higher
    root levels must be requested explicitly by the evaluator that owns the
    corresponding immutable execution SHA.
    """
    if not str(expected_execution_sha).strip():
        raise ValueError("expected immutable execution SHA is required")
    required_roots = _validated_root_level(expected_root_level)
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
    if int(final.get("roots", -1)) != required_roots:
        raise ValueError(
            f"finalized policy root level mismatch: expected {required_roots}, "
            f"got {final.get('roots')!r}"
        )
    if int(final.get("average_policy_optimizer_steps", -1)) != POLICY_STEPS:
        raise ValueError("final report AveragePolicy step count mismatch")
    if bool(final.get("strategic_selection_permitted_at_160")):
        raise ValueError("R7.5.4 artifact illegally permits strategic selection at 160 roots")
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
