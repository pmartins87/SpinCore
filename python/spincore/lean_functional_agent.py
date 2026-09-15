from __future__ import annotations

"""Offline inference agent for the first functional SpinCore checkpoint.

This module closes the training/inference semantic loop inside the simulator:
policy checkpoints are loaded into the same SPNNIV1 network family used during
training, legal actions come from the same legacy-faithful lean C++ resolver,
and the selected slot can be resolved/applied through that exact resolver.
"""

from pathlib import Path
import random
from typing import Any

import torch

from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_solver_actions import lean_legal_actions, resolve_lean_exact
from spincore.r7_5_action_cfr import legal_mask
from spincore_nn.action_models import collate_action_observations, make_policy_action_model

CHECKPOINT_SCHEMA = "SPINCORE_LEAN_FUNCTIONAL_TRAINING_V1"
REPRESENTATION = "C0_V1_FROZEN_CONTROL"
DOMAIN_BY_ID = {0: "THREE_HANDED", 1: "TRUE_HEADS_UP"}


def _street_from_state(state) -> int:
    payload = state.neural_bytes_v2()
    if len(payload) != 830 or not payload.startswith(b"SPNNIV2\x00"):
        raise RuntimeError("lean functional inference requires valid SPNNIV2 metadata")
    street = int(payload[112])
    if street not in (0, 1, 2, 3):
        raise RuntimeError(f"invalid street id {street}")
    return street


class LeanFunctionalAgent:
    def __init__(self, models: dict[str, Any], *, device: str = "cpu", seed: int = 0):
        missing = set(DOMAIN_BY_ID.values()) - set(models)
        if missing:
            raise ValueError(f"missing domain models: {sorted(missing)}")
        self.models = dict(models)
        self.device = str(device)
        self.rng = random.Random(int(seed))
        for model in self.models.values():
            model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
        seed: int = 0,
    ) -> "LeanFunctionalAgent":
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        if payload.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("wrong lean functional checkpoint schema")
        if payload.get("representation") != REPRESENTATION:
            raise ValueError("lean functional checkpoint representation drift")
        if payload.get("action_candidate") != FIRST_RELEASE_ACTION_SPEC.candidate_id:
            raise ValueError("lean functional checkpoint action-scope drift")
        if not bool(payload.get("finalized")):
            raise ValueError("inference requires a finalized average-policy checkpoint")

        domain_states = dict(payload.get("domains") or {})
        models: dict[str, Any] = {}
        for domain in DOMAIN_BY_ID.values():
            if domain not in domain_states:
                raise ValueError(f"checkpoint missing domain {domain}")
            _, model = make_policy_action_model(
                REPRESENTATION,
                device=device,
                seed=0,
            )
            model.load_state_dict(domain_states[domain]["policy"])
            model.eval()
            models[domain] = model
        return cls(models, device=device, seed=seed)

    @staticmethod
    def domain_for_state(state) -> str:
        try:
            return DOMAIN_BY_ID[int(state.domain)]
        except KeyError as exc:
            raise RuntimeError(f"unsupported solver domain id {state.domain}") from exc

    def distribution(self, state) -> tuple[int, tuple[int, ...], tuple[float, ...]]:
        if state.terminal:
            raise ValueError("cannot infer action on terminal state")
        street = _street_from_state(state)
        active_mask = FIRST_RELEASE_ACTION_SPEC.active_mask(street)
        legal = lean_legal_actions(state, active_mask)
        if not legal:
            raise RuntimeError("nonterminal lean state has no legal action")

        observation = state.neural_bytes()
        batch = collate_action_observations(
            REPRESENTATION,
            [observation],
            [legal_mask(legal)],
            device=self.device,
        )
        model = self.models[self.domain_for_state(state)]
        with torch.no_grad():
            probs = model.probabilities(batch)[0].detach().cpu().tolist()
        out = tuple(float(x) for x in probs)
        total = sum(out[action] for action in legal)
        if not (0.999 <= total <= 1.001):
            raise RuntimeError(f"average-policy probability mass drift: {total}")
        if any(out[action] < 0.0 for action in legal):
            raise RuntimeError("negative average-policy probability")
        return active_mask, legal, out

    def choose_slot(self, state, *, greedy: bool = False) -> int:
        _active_mask, legal, probs = self.distribution(state)
        if greedy:
            return max(legal, key=lambda action: probs[action])
        x = self.rng.random()
        cumulative = 0.0
        for action in legal:
            cumulative += probs[action]
            if x < cumulative:
                return int(action)
        return int(legal[-1])

    def choose_exact(self, state, *, greedy: bool = False) -> dict[str, int]:
        active_mask, _legal, _probs = self.distribution(state)
        slot = self.choose_slot(state, greedy=greedy)
        action_type, amount_to = resolve_lean_exact(state, active_mask, slot)
        return {
            "slot": int(slot),
            "action_type": int(action_type),
            "amount_to": int(amount_to),
            "active_mask": int(active_mask),
        }
