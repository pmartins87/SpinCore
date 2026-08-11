from __future__ import annotations

from copy import deepcopy
from typing import Iterable, Mapping, Sequence

import torch

from spincore_nn import AdvantageNet, NetworkConfig


SCHEMA = "SPINCORE_R7_CANDIDATE_BEHAVIOR_V1"


def _cloned_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return an immutable-by-convention CPU clone suitable for checkpoint extra data."""
    return {
        str(name): tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _state_dict_equal(model: torch.nn.Module, expected: Mapping[str, torch.Tensor]) -> bool:
    actual = model.state_dict()
    if set(actual) != set(expected):
        return False
    for name, tensor in actual.items():
        other = expected[name]
        if tensor.shape != other.shape or tensor.dtype != other.dtype:
            return False
        if not torch.equal(tensor.detach().cpu(), other.detach().cpu()):
            return False
    return True


def pack_candidate_behavior(
    *,
    kind: str,
    primary_model: torch.nn.Module,
    current_models: Sequence[torch.nn.Module],
    previous_models: Sequence[torch.nn.Module] = (),
    params: Mapping[str, object] | None = None,
    fit_generation: int | None = None,
) -> dict:
    """Serialize non-authoritative behavior-wrapper state into checkpoint ``extra``.

    The authoritative R7 checkpoint already stores ``bundle.advantage``. Candidate
    ensemble semantics add side members and, for temporal blending, a previous
    ensemble. This helper records those states without modifying the recovered
    ``SPINCORE_R7_CHECKPOINT_V2`` schema.

    ``current_models[0]`` must be the exact authoritative primary model object so
    a restored wrapper can reuse the model loaded by ``load_checkpoint`` rather
    than silently creating a second member-zero network.
    """
    if not str(kind).strip():
        raise ValueError("candidate behavior kind is required")
    if not current_models:
        raise ValueError("candidate behavior requires at least one current model")
    if current_models[0] is not primary_model:
        raise ValueError("current_models[0] must be the authoritative primary model")
    if fit_generation is not None and int(fit_generation) < 0:
        raise ValueError("fit_generation must be nonnegative")

    return {
        "schema": SCHEMA,
        "kind": str(kind),
        "current_member_count": int(len(current_models)),
        "previous_member_count": int(len(previous_models)),
        "primary_state": _cloned_state_dict(primary_model),
        "current_side_states": [_cloned_state_dict(model) for model in current_models[1:]],
        "previous_states": [_cloned_state_dict(model) for model in previous_models],
        "params": deepcopy(dict(params or {})),
        "fit_generation": None if fit_generation is None else int(fit_generation),
    }


def _restore_advantage_models(
    states: Iterable[Mapping[str, torch.Tensor]],
    *,
    config: NetworkConfig,
    device: str,
) -> list[AdvantageNet]:
    """Restore side models without advancing the authoritative global torch RNG.

    Constructing a torch module initializes parameters even though the stored
    state_dict overwrites them immediately. Without the fork this hidden
    initialization would consume the global CPU RNG after ``load_checkpoint``
    restored it, making stop/restore/continue diverge from a continuous run.
    """
    out: list[AdvantageNet] = []
    for state in states:
        with torch.random.fork_rng(devices=[]):
            model = AdvantageNet(config).to(device)
        model.load_state_dict(dict(state))
        model.eval()
        out.append(model)
    return out


def restore_candidate_behavior_models(
    payload: Mapping[str, object],
    *,
    config: NetworkConfig,
    primary_model: torch.nn.Module,
    device: str = "cpu",
) -> tuple[list[torch.nn.Module], list[torch.nn.Module], dict]:
    """Restore ensemble members around an already-restored authoritative primary model.

    Returns ``(current_models, previous_models, metadata)``. The first current
    member is the exact ``primary_model`` object supplied by the caller. A hard
    equality check against the stored primary state fails closed if the base
    checkpoint and candidate-wrapper payload do not belong to the same state.

    Restoring side members is RNG-neutral: module construction is isolated from
    the global torch RNG restored by the authoritative checkpoint.
    """
    if payload.get("schema") != SCHEMA:
        raise ValueError("wrong candidate behavior checkpoint schema")

    primary_state = payload.get("primary_state")
    if not isinstance(primary_state, Mapping):
        raise ValueError("candidate behavior primary state missing")
    if not _state_dict_equal(primary_model, primary_state):
        raise ValueError("candidate behavior primary model does not match base checkpoint")

    side_states = payload.get("current_side_states")
    previous_states = payload.get("previous_states")
    if not isinstance(side_states, Sequence) or not isinstance(previous_states, Sequence):
        raise ValueError("candidate behavior member states missing")

    current = [primary_model]
    current.extend(_restore_advantage_models(side_states, config=config, device=device))
    previous = _restore_advantage_models(previous_states, config=config, device=device)

    expected_current = int(payload.get("current_member_count", -1))
    expected_previous = int(payload.get("previous_member_count", -1))
    if len(current) != expected_current or len(previous) != expected_previous:
        raise ValueError("candidate behavior member count mismatch")

    metadata = {
        "kind": str(payload.get("kind", "")),
        "params": deepcopy(dict(payload.get("params") or {})),
        "fit_generation": payload.get("fit_generation"),
        "schema": SCHEMA,
    }
    if not metadata["kind"]:
        raise ValueError("candidate behavior kind missing")
    return current, previous, metadata
