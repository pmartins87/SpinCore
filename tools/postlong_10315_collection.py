#!/usr/bin/env python3
from __future__ import annotations

"""Memory-safe exact strong-state collector for postlong 10315 work.

PROJECT_CONTRACT_IDS:
PERF-001,PERF-002,PERF-010,PERF-013,PERF-014,PERF-015,PERF-017,
TRAIN-021,MODEL-021,MODEL-022,MODEL-023,RNG-001,RNG-002,RNG-003,
VALID-020,VALID-021,VALID-022,CKPT-001,ART-001

The canonical postlong collector keeps every decision sample and later filters
strong/Fold-legal states.  For specialist build/validation we need only those
strong samples.  This helper executes the identical sampler, solver trajectory,
teacher inference and action RNG stream, but stores only strong samples in the
same TRAIN/HOLD split.  Concatenating train_strong + hold_strong and then using
the existing unique_strong() therefore preserves the original ordering and
deduplication semantics exactly while avoiding memory proportional to every
decision.
"""

import hashlib
import json
import random
import struct
from typing import Any

import torch

import audit_3h_fresh_semantic_strategy_distill_10105 as distill
import audit_3h_semantic_fold_logit_calibrated_specialist_10115 as cal


def collect_strong_split(
    solver,
    models,
    episodes: int,
    *,
    master_seed: int,
    progress_prefix: str | None = None,
):
    sampler=distill.LegacyScenarioSampler(
        seed=int(master_seed),
        config=distill.LegacyScenarioConfig(),
    )
    action_rng=random.Random(int(master_seed)^0xAC710)
    train_strong=[]
    hold_strong=[]
    stats={
        "episodes":int(episodes),
        "train_episodes":0,
        "holdout_episodes":0,
        "decisions":0,
        "train_samples":0,
        "holdout_samples":0,
        "strong_train_samples":0,
        "strong_holdout_samples":0,
    }
    for ep_idx in range(int(episodes)):
        ep=sampler.sample_episode(force_domain=distill.DOMAIN)
        raw=solver.create(
            ep,
            int(distill.mix64(int(master_seed),ep_idx,0xDEC4)),
        )
        state=distill.LeanSolverState(raw)
        is_hold=(ep_idx%5)==0
        if is_hold:
            stats["holdout_episodes"]+=1
        else:
            stats["train_episodes"]+=1
        try:
            while not state.terminal:
                street=int(state.inner.neural_bytes_v2()[112])
                active=distill.FIRST_RELEASE_ACTION_SPEC.active_mask(street)
                legal=tuple(
                    int(x)
                    for x in state.universal_legal_actions(active)
                )
                obs=state.neural_bytes()
                sigma=distill.semantic_sigma(models,obs,legal)
                sample=distill.ActionStrategySample(
                    observation=obs,
                    legal=distill.legal_mask(legal),
                    target=tuple(float(x) for x in sigma),
                    weight=1.0,
                    iteration=10106,
                )
                stats["decisions"]+=1
                if is_hold:
                    stats["holdout_samples"]+=1
                else:
                    stats["train_samples"]+=1
                if cal.strong(sample):
                    if is_hold:
                        hold_strong.append(sample)
                        stats["strong_holdout_samples"]+=1
                    else:
                        train_strong.append(sample)
                        stats["strong_train_samples"]+=1
                action=distill.sample_action(sigma,legal,action_rng)
                distill.apply_lean(state.inner,active,action)
        finally:
            state.close()
        if progress_prefix and (ep_idx+1)%5000==0:
            print(
                f"{progress_prefix} {ep_idx+1}/{episodes} "
                f"decisions={stats['decisions']} "
                f"strong={stats['strong_train_samples']+stats['strong_holdout_samples']}",
                flush=True,
            )
    return train_strong,hold_strong,stats


def unique_strong_from_split(train_strong,hold_strong):
    return cal.unique_strong(list(train_strong)+list(hold_strong))


def canonical_collection_stats(stats):
    return {
        key:int(stats[key])
        for key in (
            "episodes",
            "train_episodes",
            "holdout_episodes",
            "decisions",
            "train_samples",
            "holdout_samples",
        )
    }


def sample_digest(samples) -> str:
    h=hashlib.sha256()
    for s in samples:
        h.update(bytes(s.observation))
        h.update(bytes(1 if bool(x) else 0 for x in s.legal))
        for value in s.target:
            h.update(struct.pack("<d",float(value)))
        h.update(struct.pack("<d",float(s.weight)))
        h.update(struct.pack("<q",int(s.iteration)))
    return h.hexdigest()


def stats_compatible_with_full(full_stats: dict[str,Any], strong_stats: dict[str,Any]) -> bool:
    keys=("episodes","train_episodes","holdout_episodes","decisions","train_samples","holdout_samples")
    return all(int(full_stats[k])==int(strong_stats[k]) for k in keys)


if __name__=="__main__":
    raise SystemExit("Import this helper from guarded postlong tools.")
