from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from spincore.r7_5_action_cfr import legal_mask, validate_policy
from spincore.r7_5_representation_v3 import H2_FINAL, H3_FINAL
from spincore.r7_5_representation_v3_checkpoint import SCHEMA as CHECKPOINT_SCHEMA
from spincore.r7_5_representation_v3_stage import FINAL_REPORT_SCHEMA
from spincore.r7_5_representation_v3_stage_contract import (
    ACTION_CANDIDATE,
    ITERATIONS,
    MODEL_FINGERPRINTS,
    MODEL_PARAMETER_COUNTS,
    POLICY_STEPS,
    TRAINING_SEEDS,
    validate_phase2_v3_contract,
)
from spincore_nn.models_v3_final import (
    collate_v3_observations,
    make_h2_final_v3,
    make_h3_final_v3,
)

LIGHT_SCHEMA = "SPINCORE_R7_5_3C_FINAL_V3_POLICY_LIGHT_V1"
DEFAULT_BATCH_SIZE = 256


def _factory(representation: str):
    if representation == H2_FINAL:
        return make_h2_final_v3, False
    if representation == H3_FINAL:
        return make_h3_final_v3, True
    raise ValueError("unsupported final V3 representation")


def extract_final_v3_policy_light(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    expected_training_execution_sha: str,
) -> dict:
    if not str(expected_training_execution_sha).strip():
        raise ValueError("training execution SHA is required")
    torch_rng = torch.get_rng_state().clone()
    try:
        payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    finally:
        torch.set_rng_state(torch_rng)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("not a final Phase2 V3 checkpoint")
    if payload.get("execution_sha") != str(expected_training_execution_sha):
        raise ValueError("training execution SHA mismatch during light extraction")
    representation = str(payload.get("representation"))
    domain = str(payload.get("domain"))
    training_seed = int(payload.get("seed", -1))
    if representation not in (H2_FINAL, H3_FINAL):
        raise ValueError("unknown final representation")
    if training_seed not in TRAINING_SEEDS:
        raise ValueError("unexpected final training seed")
    if payload.get("action_candidate") != ACTION_CANDIDATE:
        raise ValueError("final checkpoint action candidate mismatch")
    if payload.get("architecture_fingerprint_sha256") != MODEL_FINGERPRINTS[representation]:
        raise ValueError("final checkpoint architecture fingerprint mismatch")
    if bool(payload.get("production_training_authorized")) or bool(payload.get("ready_for_tables")):
        raise ValueError("final checkpoint illegally authorizes production/table use")

    progress = dict(payload.get("progress") or {})
    if progress.get("phase") != "post_policy_fit":
        raise ValueError("checkpoint is not finalized after AveragePolicy fit")
    if int(progress.get("iteration", -1)) != ITERATIONS:
        raise ValueError("final checkpoint iteration mismatch")
    if int(progress.get("policy_optimizer_step", -1)) != POLICY_STEPS:
        raise ValueError("final checkpoint policy optimizer step mismatch")
    extra = dict(payload.get("extra") or {})
    final_report = dict(extra.get("final_report") or {})
    if final_report.get("schema") != FINAL_REPORT_SCHEMA:
        raise ValueError("final checkpoint missing final report")
    if final_report.get("representation") != representation:
        raise ValueError("final report representation mismatch")
    if final_report.get("domain") != domain:
        raise ValueError("final report domain mismatch")
    if int(final_report.get("training_seed", -1)) != training_seed:
        raise ValueError("final report training seed mismatch")
    if int(final_report.get("iterations", -1)) != ITERATIONS:
        raise ValueError("final report iteration count mismatch")
    if int(final_report.get("roots", -1)) != ITERATIONS * 64:
        raise ValueError("final report root count mismatch")
    if int(final_report.get("average_policy_optimizer_steps", -1)) != POLICY_STEPS:
        raise ValueError("final report policy optimizer step mismatch")

    light = {
        "schema": LIGHT_SCHEMA,
        "representation": representation,
        "domain": domain,
        "training_seed": training_seed,
        "action_candidate": ACTION_CANDIDATE,
        "training_execution_sha": str(expected_training_execution_sha),
        "architecture_fingerprint_sha256": MODEL_FINGERPRINTS[representation],
        "config": dict(payload["config"]),
        "parameter_count": MODEL_PARAMETER_COUNTS[representation],
        "policy_state_dict": payload["policy"],
        "final_report": final_report,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(light, output)
    return {
        key: value
        for key, value in light.items()
        if key not in ("policy_state_dict",)
    }


@dataclass
class FinalizedV3Policy:
    representation: str
    domain: str
    training_seed: int
    model: object
    final_report: dict
    training_execution_sha: str
    artifact_path: str

    @property
    def with_semantics(self) -> bool:
        return self.representation == H3_FINAL

    def batch_probabilities(
        self,
        observations: Sequence[bytes],
        legal_sets: Sequence[tuple[int, ...]],
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> tuple[tuple[float, ...], ...]:
        if len(observations) != len(legal_sets):
            raise ValueError("V3 final policy observation/legal count mismatch")
        if not observations:
            return ()
        width = int(batch_size)
        if width <= 0:
            raise ValueError("positive V3 final-policy batch size required")
        out: list[tuple[float, ...]] = []
        self.model.eval()
        for start in range(0, len(observations), width):
            obs = list(observations[start : start + width])
            legal = list(legal_sets[start : start + width])
            batch = collate_v3_observations(
                obs,
                [legal_mask(row) for row in legal],
                with_semantics=self.with_semantics,
                device="cpu",
            )
            with torch.no_grad():
                logits = self.model(batch).masked_fill(~batch["legal"], -1e9)
                probabilities = torch.softmax(logits, dim=-1).detach().cpu().tolist()
            for raw, legal_row in zip(probabilities, legal):
                out.append(validate_policy(tuple(float(value) for value in raw), tuple(legal_row)))
        if len(out) != len(observations):
            raise RuntimeError("V3 final policy batch output count drift")
        return tuple(out)

    def __call__(self, _state, observation: bytes, legal: tuple[int, ...]) -> tuple[float, ...]:
        return self.batch_probabilities([observation], [legal], batch_size=1)[0]


def load_finalized_v3_policy_light(
    artifact_path: str | Path,
    *,
    repo_root: str | Path,
    expected_training_execution_sha: str,
    expected_representation: str | None = None,
    expected_domain: str | None = None,
    expected_training_seed: int | None = None,
) -> FinalizedV3Policy:
    torch_rng = torch.get_rng_state().clone()
    try:
        payload = torch.load(Path(artifact_path), map_location="cpu", weights_only=False)
    finally:
        torch.set_rng_state(torch_rng)
    if payload.get("schema") != LIGHT_SCHEMA:
        raise ValueError("wrong final V3 light-policy schema")
    representation = str(payload.get("representation"))
    domain = str(payload.get("domain"))
    training_seed = int(payload.get("training_seed", -1))
    if payload.get("training_execution_sha") != str(expected_training_execution_sha):
        raise ValueError("final V3 light-policy execution SHA mismatch")
    if expected_representation is not None and representation != str(expected_representation):
        raise ValueError("final V3 light-policy representation mismatch")
    if expected_domain is not None and domain != str(expected_domain):
        raise ValueError("final V3 light-policy domain mismatch")
    if expected_training_seed is not None and training_seed != int(expected_training_seed):
        raise ValueError("final V3 light-policy training seed mismatch")

    contract = validate_phase2_v3_contract(
        repo_root,
        representation=representation,
        domain=domain,
        training_seed=training_seed,
    )
    if payload.get("architecture_fingerprint_sha256") != MODEL_FINGERPRINTS[representation]:
        raise ValueError("final V3 light-policy fingerprint mismatch")
    if int(payload.get("parameter_count", -1)) != MODEL_PARAMETER_COUNTS[representation]:
        raise ValueError("final V3 light-policy parameter-count mismatch")
    if dict(payload.get("config") or {}) != contract["live_model"]["config"]:
        raise ValueError("final V3 light-policy config mismatch")
    if bool(payload.get("production_training_authorized")) or bool(payload.get("ready_for_tables")):
        raise ValueError("light policy illegally authorizes production/table use")

    factory, _with_semantics = _factory(representation)
    _, model = factory(device="cpu", seed=0)
    model.load_state_dict(payload["policy_state_dict"])
    model.eval()
    if not torch.equal(torch_rng, torch.get_rng_state()):
        raise RuntimeError("loading final V3 light policy changed global Torch RNG")
    return FinalizedV3Policy(
        representation=representation,
        domain=domain,
        training_seed=training_seed,
        model=model,
        final_report=dict(payload.get("final_report") or {}),
        training_execution_sha=str(expected_training_execution_sha),
        artifact_path=str(Path(artifact_path)),
    )
