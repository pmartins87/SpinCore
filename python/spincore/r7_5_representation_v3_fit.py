from __future__ import annotations

import math
import random
from typing import Sequence

import torch

from spincore.r7 import stratified_audit_indices, weighted_mean_tv, weighted_normalized_rmse
from spincore.r7_5_representation_v3 import H2_FINAL, H3_FINAL
from spincore_nn.models_v3_final import (
    collate_v3_observations,
    make_h2_final_v3,
    make_h3_final_v3,
)
from spincore_nn.training import train_step


def _with_semantics(representation: str) -> bool:
    if representation == H2_FINAL:
        return False
    if representation == H3_FINAL:
        return True
    raise ValueError(f"unsupported final V3 representation {representation!r}")


def _make_model(representation: str, *, device: str, seed: int):
    if representation == H2_FINAL:
        return make_h2_final_v3(device=device, seed=int(seed))
    if representation == H3_FINAL:
        return make_h3_final_v3(device=device, seed=int(seed))
    raise ValueError(f"unsupported final V3 representation {representation!r}")


def _batch(representation: str, samples: Sequence, *, device: str):
    batch = collate_v3_observations(
        [sample.observation for sample in samples],
        [sample.legal for sample in samples],
        with_semantics=_with_semantics(representation),
        device=device,
    )
    target = torch.tensor(
        [sample.target for sample in samples], dtype=torch.float32, device=device
    )
    weights = torch.tensor(
        [sample.weight for sample in samples], dtype=torch.float32, device=device
    )
    return batch, target, weights


def _policy_probabilities(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits = model(batch).masked_fill(~batch["legal"], -1e9)
        return torch.softmax(logits, dim=-1).detach().cpu()


def audit_v3_advantage_model(
    model,
    memory_items: Sequence,
    *,
    representation: str,
    sample_size: int,
    seed: int,
    device: str = "cpu",
) -> float:
    indices = stratified_audit_indices(len(memory_items), int(sample_size), int(seed))
    if not indices:
        return math.inf
    samples = [memory_items[index] for index in indices]
    batch, target, weights = _batch(representation, samples, device=device)
    model.eval()
    with torch.no_grad():
        pred = model(batch)
    return weighted_normalized_rmse(pred, target, batch["legal"], weights)


def audit_v3_policy_model(
    model,
    memory_items: Sequence,
    *,
    representation: str,
    sample_size: int,
    seed: int,
    device: str = "cpu",
) -> float:
    indices = stratified_audit_indices(len(memory_items), int(sample_size), int(seed))
    if not indices:
        return math.inf
    samples = [memory_items[index] for index in indices]
    batch, target, weights = _batch(representation, samples, device=device)
    pred = _policy_probabilities(model, batch)
    return weighted_mean_tv(pred, target.detach().cpu(), weights.detach().cpu())


def ensemble_v3_advantage_nrmse(
    models: Sequence,
    memory_items: Sequence,
    *,
    representation: str,
    sample_size: int,
    seed: int,
    device: str = "cpu",
) -> float:
    if not models:
        return math.inf
    indices = stratified_audit_indices(len(memory_items), int(sample_size), int(seed))
    if not indices:
        return math.inf
    samples = [memory_items[index] for index in indices]
    batch, target, weights = _batch(representation, samples, device=device)
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            predictions.append(model(batch).detach())
    mean_prediction = torch.stack(predictions, dim=0).mean(dim=0)
    return weighted_normalized_rmse(mean_prediction, target, batch["legal"], weights)


def fit_independent_v3_advantage_member(
    memory_items: Sequence,
    *,
    representation: str,
    init_seed: int,
    batch_seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    device: str = "cpu",
) -> tuple[object, dict]:
    items = list(memory_items)
    if not items:
        raise ValueError("cannot fit V3 advantage member from empty memory")
    if steps < 0 or batch_size <= 0 or learning_rate <= 0.0:
        raise ValueError("invalid V3 independent member fit contract")
    _, model = _make_model(
        representation,
        device=device,
        seed=int(init_seed),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    rng = random.Random(int(batch_seed))
    losses: list[float] = []
    count = min(int(batch_size), len(items))
    for _ in range(int(steps)):
        samples = rng.sample(items, count)
        batch, target, weights = _batch(representation, samples, device=device)
        losses.append(train_step(model, optimizer, batch, target, weights, "advantage"))
    return model, {
        "init_seed": int(init_seed),
        "batch_seed": int(batch_seed),
        "steps": int(steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "mean_loss": float(sum(losses) / len(losses)) if losses else math.nan,
        "final_loss": float(losses[-1]) if losses else math.nan,
        "caller_rng_isolation": True,
        "memory_materializations": 1,
    }


def cross_seed_v3_policy_tv(
    model_a,
    model_b,
    observations: Sequence[tuple[bytes, tuple[int, ...]]],
    *,
    representation: str,
    device: str = "cpu",
) -> dict[str, float]:
    if not observations:
        return {
            "mean_tv": math.inf,
            "p50_tv": math.inf,
            "p95_tv": math.inf,
            "max_tv": math.inf,
        }
    batch = collate_v3_observations(
        [observation for observation, _ in observations],
        [legal for _, legal in observations],
        with_semantics=_with_semantics(representation),
        device=device,
    )
    a = _policy_probabilities(model_a, batch)
    b = _policy_probabilities(model_b, batch)
    tv = 0.5 * torch.abs(a - b).sum(dim=1)
    quantiles = torch.quantile(tv, torch.tensor([0.5, 0.95]))
    return {
        "mean_tv": float(tv.mean()),
        "p50_tv": float(quantiles[0]),
        "p95_tv": float(quantiles[1]),
        "max_tv": float(tv.max()),
    }
