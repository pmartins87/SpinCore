#!/usr/bin/env python3
from __future__ import annotations

"""Exact post-15315 teacher-target collectors with explicit lineage metadata.

This module preserves the same scenario sampler, chance-seed derivation,
teacher inference, action RNG, TRAIN/HOLD split and action application used by
the validated 10315 collection path. The only deliberate change is that
ActionStrategySample.iteration is supplied explicitly and is frozen to the
teacher lineage identifier (15315) by the caller instead of inheriting the
historical 10106 metadata.

PROJECT_CONTRACT_IDS:
TRAIN-023,MODEL-025,RNG-001,RNG-003,RNG-013,VALID-025,VALID-031,VALID-035
"""

import hashlib
import random
import struct
from typing import Any, Callable

import audit_3h_fresh_semantic_strategy_distill_10105 as distill
import audit_3h_semantic_fold_logit_calibrated_specialist_10115 as cal


def _episode_stream(
    solver,
    models,
    episodes: int,
    *,
    master_seed: int,
    sample_iteration: int,
    keep: Callable[[Any], bool] | None,
    progress_prefix: str | None,
):
    sampler = distill.LegacyScenarioSampler(
        seed=int(master_seed),
        config=distill.LegacyScenarioConfig(),
    )
    action_rng = random.Random(int(master_seed) ^ 0xAC710)
    train = []
    hold = []
    stats = {
        "episodes": int(episodes),
        "train_episodes": 0,
        "holdout_episodes": 0,
        "decisions": 0,
        "train_samples": 0,
        "holdout_samples": 0,
        "kept_train_samples": 0,
        "kept_holdout_samples": 0,
        "sample_iteration": int(sample_iteration),
    }
    for ep_idx in range(int(episodes)):
        ep = sampler.sample_episode(force_domain=distill.DOMAIN)
        raw = solver.create(
            ep,
            int(distill.mix64(int(master_seed), ep_idx, 0xDEC4)),
        )
        state = distill.LeanSolverState(raw)
        is_hold = (ep_idx % 5) == 0
        if is_hold:
            stats["holdout_episodes"] += 1
        else:
            stats["train_episodes"] += 1
        try:
            while not state.terminal:
                street = int(state.inner.neural_bytes_v2()[112])
                active = distill.FIRST_RELEASE_ACTION_SPEC.active_mask(street)
                legal = tuple(
                    int(x) for x in state.universal_legal_actions(active)
                )
                obs = state.neural_bytes()
                sigma = distill.semantic_sigma(models, obs, legal)
                sample = distill.ActionStrategySample(
                    observation=obs,
                    legal=distill.legal_mask(legal),
                    target=tuple(float(x) for x in sigma),
                    weight=1.0,
                    iteration=int(sample_iteration),
                )
                stats["decisions"] += 1
                if is_hold:
                    stats["holdout_samples"] += 1
                else:
                    stats["train_samples"] += 1

                if keep is None or bool(keep(sample)):
                    if is_hold:
                        hold.append(sample)
                        stats["kept_holdout_samples"] += 1
                    else:
                        train.append(sample)
                        stats["kept_train_samples"] += 1

                action = distill.sample_action(sigma, legal, action_rng)
                distill.apply_lean(state.inner, active, action)
        finally:
            state.close()

        if progress_prefix and (ep_idx + 1) % 5000 == 0:
            print(
                f"{progress_prefix} {ep_idx + 1}/{episodes} "
                f"decisions={stats['decisions']} "
                f"kept={stats['kept_train_samples'] + stats['kept_holdout_samples']}",
                flush=True,
            )
    return train, hold, stats


def collect_fresh_split(
    solver,
    models,
    episodes: int,
    *,
    master_seed: int,
    sample_iteration: int,
    progress_prefix: str | None = None,
):
    """Collect every decision with explicit teacher-lineage metadata."""
    return _episode_stream(
        solver,
        models,
        episodes,
        master_seed=master_seed,
        sample_iteration=sample_iteration,
        keep=None,
        progress_prefix=progress_prefix,
    )


def collect_strong_split(
    solver,
    models,
    episodes: int,
    *,
    master_seed: int,
    sample_iteration: int,
    progress_prefix: str | None = None,
):
    """Collect only the canonical postflop strong/Fold-legal subset."""
    return _episode_stream(
        solver,
        models,
        episodes,
        master_seed=master_seed,
        sample_iteration=sample_iteration,
        keep=cal.strong,
        progress_prefix=progress_prefix,
    )


def unique_strong_from_split(train_strong, hold_strong):
    return cal.unique_strong(list(train_strong) + list(hold_strong))


def canonical_collection_stats(stats: dict[str, Any]) -> dict[str, int]:
    return {
        key: int(stats[key])
        for key in (
            "episodes",
            "train_episodes",
            "holdout_episodes",
            "decisions",
            "train_samples",
            "holdout_samples",
            "kept_train_samples",
            "kept_holdout_samples",
            "sample_iteration",
        )
    }


def sample_digest(samples) -> str:
    h = hashlib.sha256()
    for s in samples:
        h.update(bytes(s.observation))
        h.update(bytes(1 if bool(x) else 0 for x in s.legal))
        for value in s.target:
            h.update(struct.pack("<d", float(value)))
        h.update(struct.pack("<d", float(s.weight)))
        h.update(struct.pack("<q", int(s.iteration)))
    return h.hexdigest()


if __name__ == "__main__":
    raise SystemExit("Import this helper from guarded post-15315 tools.")
