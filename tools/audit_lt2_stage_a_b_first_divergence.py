#!/usr/bin/env python3
from __future__ import annotations

"""Forensic Stage-A -> Stage-B AveragePolicy first-divergence attribution.

This read-only audit replays the already-seen weak-baseline diagnostic seeds on
paired HU scenarios. Stage A and Stage B receive the same scenario, deal, weak
opponent, hero seat, and random streams. Until their sampled hero actions first
differ, the two solver states are identical.

Every seat-run is assigned exactly one first-divergence group:
  NO_DIVERGENCE
  PREFLOP_ROOT
  PREFLOP_FACING_ALL_IN
  PREFLOP_OTHER
  FLOP
  TURN
  RIVER

The B-minus-A chip-EV contribution of those mutually exclusive groups therefore
adds back to the total paired B-minus-A result.

This localizes where the deployed AveragePolicy regression manifests. It does
not by itself prove that any Advantage-target mechanism caused that regression.
"""

import argparse
from dataclasses import asdict
import json
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import random
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler  # noqa: E402
from spincore.solver import Episode  # noqa: E402

BASELINES = ("UNIFORM_LEGAL", "PASSIVE_CALLER", "JAMMER")
GROUPS = (
    "NO_DIVERGENCE",
    "PREFLOP_ROOT",
    "PREFLOP_FACING_ALL_IN",
    "PREFLOP_OTHER",
    "FLOP",
    "TURN",
    "RIVER",
)

_SOLVER = None
_AGENT_A = None
_AGENT_B = None

POLICY_SNAPSHOT_SCHEMA = "SPINCORE_LT2_HU_POLICY_SNAPSHOT_V1"
REPRESENTATION = "C0_V1_FROZEN_CONTROL"


class _HUPolicyAgent:
    """Minimal TRUE_HEADS_UP AveragePolicy inference wrapper."""

    def __init__(self, model):
        self.model = model
        self.model.eval()

    def distribution(self, state):
        import torch
        from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
        from spincore.lean_functional_agent import _street_from_state
        from spincore.lean_solver_actions import lean_legal_actions
        from spincore.r7_5_action_cfr import legal_mask
        from spincore_nn.action_models import collate_action_observations

        if state.terminal:
            raise ValueError("cannot infer action on terminal state")
        street = int(_street_from_state(state))
        active_mask = FIRST_RELEASE_ACTION_SPEC.active_mask(street)
        legal = lean_legal_actions(state, active_mask)
        if not legal:
            raise RuntimeError("nonterminal HU state has no legal action")
        observation = state.neural_bytes()
        batch = collate_action_observations(
            REPRESENTATION,
            [observation],
            [legal_mask(legal)],
            device="cpu",
        )
        with torch.no_grad():
            probs = self.model.probabilities(batch)[0].detach().cpu().tolist()
        out = tuple(float(x) for x in probs)
        total = sum(out[action] for action in legal)
        if not (0.999 <= total <= 1.001):
            raise RuntimeError(f"HU policy probability mass drift: {total}")
        return int(active_mask), tuple(int(x) for x in legal), out


def _torch_load_mmap(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        return torch.load(path, map_location="cpu", weights_only=False)


def _extract_hu_policy_snapshot(checkpoint: Path, destination: Path) -> dict[str, Any]:
    """Extract only the HU AveragePolicy once in the parent process.

    The full training checkpoint contains large reservoirs. Loading that entire
    object independently in every spawned worker is unnecessary and can exhaust
    RAM. The derived snapshot is tiny and contains only the deployed HU policy.
    """
    import gc
    import torch

    payload = _torch_load_mmap(checkpoint)
    if payload.get("schema") != "SPINCORE_LEAN_FUNCTIONAL_TRAINING_V1":
        raise RuntimeError(f"unexpected checkpoint schema: {checkpoint}")
    if payload.get("representation") != REPRESENTATION:
        raise RuntimeError(f"representation drift in {checkpoint}")
    if not bool(payload.get("finalized")):
        raise RuntimeError(f"forensic source is not finalized: {checkpoint}")
    domains = dict(payload.get("domains") or {})
    if "TRUE_HEADS_UP" not in domains:
        raise RuntimeError(f"checkpoint missing TRUE_HEADS_UP: {checkpoint}")

    policy = {
        key: value.detach().cpu().clone()
        for key, value in domains["TRUE_HEADS_UP"]["policy"].items()
    }
    snapshot = {
        "schema": POLICY_SNAPSHOT_SCHEMA,
        "source_checkpoint": str(checkpoint.resolve()),
        "completed_iteration": int(payload["completed_iteration"]),
        "representation": str(payload["representation"]),
        "action_candidate": payload.get("action_candidate"),
        "policy": policy,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(snapshot, destination)
    meta = {
        "completed_iteration": int(snapshot["completed_iteration"]),
        "snapshot_path": str(destination.resolve()),
        "parameter_tensors": len(policy),
        "snapshot_bytes": int(destination.stat().st_size),
    }
    del policy, snapshot, domains, payload
    gc.collect()
    return meta


def _load_hu_policy_snapshot(path: str):
    import torch
    from spincore_nn.action_models import make_policy_action_model

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema") != POLICY_SNAPSHOT_SCHEMA:
        raise RuntimeError(f"wrong HU policy snapshot schema: {path}")
    if payload.get("representation") != REPRESENTATION:
        raise RuntimeError(f"HU policy snapshot representation drift: {path}")
    _, model = make_policy_action_model(
        REPRESENTATION,
        device="cpu",
        seed=0,
    )
    model.load_state_dict(payload["policy"])
    model.eval()
    return _HUPolicyAgent(model)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver", type=Path, default=ROOT / "build/libspincore_solver_c.so")
    p.add_argument("--stage-a", type=Path, required=True)
    p.add_argument("--stage-b", type=Path, required=True)
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[20260920, 20260921, 20260922, 20260923, 20260924, 20260925],
    )
    p.add_argument("--scenarios-per-seed", type=int, default=5000)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--report", type=Path, required=True)
    return p.parse_args()


def _mix64(*values: int) -> int:
    x = 0x9E3779B97F4A7C15
    mask = (1 << 64) - 1
    for value in values:
        y = int(value) & mask
        x ^= (y + 0x9E3779B97F4A7C15 + ((x << 6) & mask) + (x >> 2)) & mask
        x &= mask
    return x


def _init_worker(
    solver_path: str,
    stage_a_policy_snapshot: str,
    stage_b_policy_snapshot: str,
) -> None:
    global _SOLVER, _AGENT_A, _AGENT_B
    import os
    import torch
    from spincore.solver import SolverLibrary

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["SPINCORE_TORCH_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    _SOLVER = SolverLibrary(solver_path)
    _AGENT_A = _load_hu_policy_snapshot(stage_a_policy_snapshot)
    _AGENT_B = _load_hu_policy_snapshot(stage_b_policy_snapshot)


def _street(state) -> int:
    from spincore.lean_functional_agent import _street_from_state

    return int(_street_from_state(state))


def _legal_context(state):
    from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
    from spincore.lean_solver_actions import lean_legal_actions

    street = _street(state)
    active_mask = FIRST_RELEASE_ACTION_SPEC.active_mask(street)
    legal = lean_legal_actions(state, active_mask)
    if not legal:
        raise RuntimeError("nonterminal state has no lean legal action")
    return int(active_mask), tuple(int(x) for x in legal)


def _sample_probs(
    legal: tuple[int, ...],
    probs: tuple[float, ...],
    rng: random.Random,
) -> int:
    x = rng.random()
    cumulative = 0.0
    for slot in legal:
        cumulative += float(probs[slot])
        if x < cumulative:
            return int(slot)
    return int(legal[-1])


def _baseline_action(
    baseline: str,
    state,
    rng: random.Random,
) -> tuple[int, int]:
    active_mask, legal = _legal_context(state)
    if baseline == "UNIFORM_LEGAL":
        return active_mask, int(legal[rng.randrange(len(legal))])
    if baseline == "PASSIVE_CALLER":
        if 1 in legal:
            return active_mask, 1
        if 0 in legal:
            return active_mask, 0
        return active_mask, int(min(legal))
    if baseline == "JAMMER":
        if 9 in legal:
            return active_mask, 9
        if 1 in legal:
            return active_mask, 1
        if 0 in legal:
            return active_mask, 0
        return active_mask, int(max(legal))
    raise ValueError(f"unknown baseline {baseline}")


def _agent_action(agent, state, rng: random.Random):
    active_mask, legal, probs = agent.distribution(state)
    slot = _sample_probs(tuple(legal), tuple(probs), rng)
    return int(active_mask), tuple(int(x) for x in legal), tuple(float(x) for x in probs), int(slot)


def _finish_arm(
    state,
    *,
    hero_seat: int,
    agent,
    baseline: str,
    seat_rng: dict[int, random.Random],
) -> int:
    from spincore.lean_solver_actions import apply_lean

    decisions = 0
    try:
        while not state.terminal:
            actor = int(state.actor)
            if actor == int(hero_seat):
                active_mask, _legal, _probs, slot = _agent_action(agent, state, seat_rng[actor])
            else:
                active_mask, slot = _baseline_action(baseline, state, seat_rng[actor])
            apply_lean(state, active_mask, slot)
            decisions += 1
            if decisions > 200:
                raise RuntimeError("forensic arm exceeded 200 decisions")
        delta = state.terminal_chip_delta()
        if sum(delta) != 0:
            raise RuntimeError(f"terminal chip delta not zero-sum: {delta}")
        return int(delta[int(hero_seat)])
    finally:
        state.close()


def _group_for(
    *,
    street: int,
    common_path: list[tuple[int, int]],
    last_actor: int | None,
    last_action: int | None,
    hero_seat: int,
) -> str:
    if street == 0:
        if not common_path:
            return "PREFLOP_ROOT"
        if last_actor is not None and last_actor != int(hero_seat) and int(last_action) == 9:
            return "PREFLOP_FACING_ALL_IN"
        return "PREFLOP_OTHER"
    return {1: "FLOP", 2: "TURN", 3: "RIVER"}[int(street)]


def _paired_run(
    episode: Episode,
    *,
    deal_seed: int,
    hero_seat: int,
    baseline: str,
    seed: int,
    scenario_index: int,
) -> dict[str, Any]:
    from spincore.lean_solver_actions import apply_lean

    state_a = _SOLVER.create(episode, int(deal_seed))
    state_b = _SOLVER.create(episode, int(deal_seed))

    rng_a = {
        seat: random.Random(_mix64(seed, scenario_index, seat, 100 + BASELINES.index(baseline)))
        for seat in range(3)
    }
    rng_b = {
        seat: random.Random(_mix64(seed, scenario_index, seat, 100 + BASELINES.index(baseline)))
        for seat in range(3)
    }
    rng_a[int(hero_seat)] = random.Random(_mix64(seed, scenario_index, hero_seat, 777))
    rng_b[int(hero_seat)] = random.Random(_mix64(seed, scenario_index, hero_seat, 777))

    common_path: list[tuple[int, int]] = []
    last_actor = None
    last_action = None
    common_hero_decisions = 0
    divergence = None

    try:
        decisions = 0
        while True:
            if bool(state_a.terminal) != bool(state_b.terminal):
                raise RuntimeError("paired states changed terminal status before hero divergence")
            if state_a.terminal:
                da = state_a.terminal_chip_delta()
                db = state_b.terminal_chip_delta()
                if tuple(da) != tuple(db):
                    raise RuntimeError("identical action path produced different terminal result")
                value = int(da[int(hero_seat)])
                state_a.close()
                state_b.close()
                state_a = None
                state_b = None
                return {
                    "stage_a": value,
                    "stage_b": value,
                    "delta_b_minus_a": 0,
                    "group": "NO_DIVERGENCE",
                    "diverged": False,
                    "street": None,
                    "last_action_slot": None,
                    "facing_all_in": False,
                    "a_slot": None,
                    "b_slot": None,
                    "policy_tv": 0.0,
                    "b_minus_a_mass_fold": 0.0,
                    "b_minus_a_mass_check_call": 0.0,
                    "b_minus_a_mass_all_in": 0.0,
                    "common_hero_decisions_before_divergence": int(common_hero_decisions),
                    "common_public_action_count": len(common_path),
                    "transition": None,
                }

            actor_a = int(state_a.actor)
            actor_b = int(state_b.actor)
            if actor_a != actor_b:
                raise RuntimeError("paired actor drift before hero divergence")
            actor = actor_a

            if actor != int(hero_seat):
                mask_a, slot_a = _baseline_action(baseline, state_a, rng_a[actor])
                mask_b, slot_b = _baseline_action(baseline, state_b, rng_b[actor])
                if mask_a != mask_b or slot_a != slot_b:
                    raise RuntimeError("baseline action drift before hero divergence")
                apply_lean(state_a, mask_a, slot_a)
                apply_lean(state_b, mask_b, slot_b)
                common_path.append((actor, int(slot_a)))
                last_actor, last_action = actor, int(slot_a)
            else:
                mask_a, legal_a, probs_a, slot_a = _agent_action(_AGENT_A, state_a, rng_a[actor])
                mask_b, legal_b, probs_b, slot_b = _agent_action(_AGENT_B, state_b, rng_b[actor])
                if mask_a != mask_b or legal_a != legal_b:
                    raise RuntimeError("A/B legal action drift on identical state")
                common_hero_decisions += 1

                if int(slot_a) != int(slot_b):
                    street = _street(state_a)
                    group = _group_for(
                        street=street,
                        common_path=common_path,
                        last_actor=last_actor,
                        last_action=last_action,
                        hero_seat=int(hero_seat),
                    )
                    tv = 0.5 * sum(
                        abs(float(probs_a[i]) - float(probs_b[i]))
                        for i in range(10)
                    )
                    divergence = {
                        "group": group,
                        "street": int(street),
                        "last_action_slot": None if last_action is None else int(last_action),
                        "facing_all_in": bool(group == "PREFLOP_FACING_ALL_IN"),
                        "a_slot": int(slot_a),
                        "b_slot": int(slot_b),
                        "policy_tv": float(tv),
                        "b_minus_a_mass_fold": float(probs_b[0] - probs_a[0]),
                        "b_minus_a_mass_check_call": float(probs_b[1] - probs_a[1]),
                        "b_minus_a_mass_all_in": float(probs_b[9] - probs_a[9]),
                        "common_hero_decisions_before_divergence": int(common_hero_decisions - 1),
                        "common_public_action_count": len(common_path),
                        "transition": f"{int(slot_a)}->{int(slot_b)}",
                    }
                    apply_lean(state_a, mask_a, slot_a)
                    apply_lean(state_b, mask_b, slot_b)
                    break

                apply_lean(state_a, mask_a, slot_a)
                apply_lean(state_b, mask_b, slot_b)
                common_path.append((actor, int(slot_a)))
                last_actor, last_action = actor, int(slot_a)

            decisions += 1
            if decisions > 200:
                raise RuntimeError("paired forensic path exceeded 200 decisions")

        if divergence is None:
            raise RuntimeError("divergence bookkeeping failure")

        value_a = _finish_arm(
            state_a,
            hero_seat=int(hero_seat),
            agent=_AGENT_A,
            baseline=baseline,
            seat_rng=rng_a,
        )
        state_a = None
        value_b = _finish_arm(
            state_b,
            hero_seat=int(hero_seat),
            agent=_AGENT_B,
            baseline=baseline,
            seat_rng=rng_b,
        )
        state_b = None
        return {
            "stage_a": int(value_a),
            "stage_b": int(value_b),
            "delta_b_minus_a": int(value_b - value_a),
            "diverged": True,
            **divergence,
        }
    finally:
        if state_a is not None:
            state_a.close()
        if state_b is not None:
            state_b.close()


def _worker(task: tuple[int, int, Episode, int]) -> list[dict[str, Any]]:
    seed, scenario_index, episode, deal_seed = task
    if not episode.game_is_hu:
        return []
    live = [seat for seat, stack in enumerate(episode.stacks) if int(stack) > 0]
    if len(live) != 2:
        raise RuntimeError("HU forensic task does not have exactly two live seats")

    blind = f"{episode.small_blind}/{episode.big_blind}"
    rows = []
    for baseline in BASELINES:
        for hero_seat in live:
            result = _paired_run(
                episode,
                deal_seed=int(deal_seed),
                hero_seat=int(hero_seat),
                baseline=baseline,
                seed=int(seed),
                scenario_index=int(scenario_index),
            )
            rows.append(
                {
                    "seed": int(seed),
                    "scenario": int(scenario_index),
                    "blind": blind,
                    "hero_seat": int(hero_seat),
                    "baseline": baseline,
                    **result,
                }
            )
    return rows


def _mean_ci(values: list[float]) -> dict[str, float | int]:
    n = len(values)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "sem": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    mean = float(statistics.fmean(values))
    sem = 0.0 if n == 1 else float(statistics.stdev(values) / math.sqrt(n))
    half = 1.96 * sem
    return {
        "n": int(n),
        "mean": mean,
        "sem": sem,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def _cluster_overall(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cluster: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        by_cluster.setdefault((int(row["seed"]), int(row["scenario"])), []).append(row)
    values = [
        statistics.fmean(float(r["delta_b_minus_a"]) for r in rr)
        for rr in by_cluster.values()
    ]
    return {
        "scenario_clusters": len(by_cluster),
        "seat_runs": len(rows),
        "b_minus_a_chip_ev": _mean_ci(values),
        "divergence_rate": float(
            statistics.fmean(1.0 if bool(r["diverged"]) else 0.0 for r in rows)
        ) if rows else float("nan"),
    }


def _group_contribution(rows: list[dict[str, Any]], group: str) -> dict[str, Any]:
    by_cluster: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        by_cluster.setdefault((int(row["seed"]), int(row["scenario"])), []).append(row)

    contributions = []
    for rr in by_cluster.values():
        denom = float(len(rr))
        contributions.append(
            sum(
                float(r["delta_b_minus_a"])
                for r in rr
                if str(r["group"]) == str(group)
            )
            / denom
        )

    selected = [r for r in rows if str(r["group"]) == str(group)]
    conditional_delta = [float(r["delta_b_minus_a"]) for r in selected]
    diverged = [r for r in selected if bool(r["diverged"])]
    out = {
        "seat_runs": len(selected),
        "seat_run_fraction": float(len(selected) / len(rows)) if rows else 0.0,
        "contribution_to_overall_b_minus_a_chip_ev": _mean_ci(contributions),
        "conditional_b_minus_a_chip_delta": _mean_ci(conditional_delta),
    }
    if diverged:
        out["first_divergence_policy_tv"] = _mean_ci(
            [float(r["policy_tv"]) for r in diverged]
        )
        out["mean_b_minus_a_probability_mass"] = {
            "FOLD": float(statistics.fmean(float(r["b_minus_a_mass_fold"]) for r in diverged)),
            "CHECK_CALL": float(
                statistics.fmean(float(r["b_minus_a_mass_check_call"]) for r in diverged)
            ),
            "ALL_IN": float(
                statistics.fmean(float(r["b_minus_a_mass_all_in"]) for r in diverged)
            ),
        }
    else:
        out["first_divergence_policy_tv"] = _mean_ci([])
        out["mean_b_minus_a_probability_mass"] = {
            "FOLD": 0.0,
            "CHECK_CALL": 0.0,
            "ALL_IN": 0.0,
        }
    return out


def _transition_summary(rows: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    subset = [r for r in rows if str(r["group"]) == group and r.get("transition")]
    transitions = sorted({str(r["transition"]) for r in subset})
    out = []
    for transition in transitions:
        rr = [r for r in subset if str(r["transition"]) == transition]
        out.append(
            {
                "transition": transition,
                "seat_runs": len(rr),
                "mean_b_minus_a_chip_delta": float(
                    statistics.fmean(float(r["delta_b_minus_a"]) for r in rr)
                ),
                "mean_policy_tv": float(
                    statistics.fmean(float(r["policy_tv"]) for r in rr)
                ),
            }
        )
    out.sort(key=lambda x: (-int(x["seat_runs"]), str(x["transition"])))
    return out


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for baseline in BASELINES:
        br = [r for r in rows if r["baseline"] == baseline]
        overall = _cluster_overall(br)
        groups = {group: _group_contribution(br, group) for group in GROUPS}
        contribution_sum = sum(
            float(groups[group]["contribution_to_overall_b_minus_a_chip_ev"]["mean"])
            for group in GROUPS
        )
        overall_mean = float(overall["b_minus_a_chip_ev"]["mean"])
        summary[baseline] = {
            "overall": overall,
            "first_divergence_groups": groups,
            "contribution_closure_abs_error": float(abs(contribution_sum - overall_mean)),
            "preflop_facing_all_in_transitions": _transition_summary(
                br, "PREFLOP_FACING_ALL_IN"
            ),
            "preflop_other_transitions": _transition_summary(br, "PREFLOP_OTHER"),
        }
    return summary


def main() -> int:
    args = parse_args()
    if args.scenarios_per_seed <= 0 or args.workers <= 0:
        raise SystemExit("positive scenarios/workers required")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("seeds must be nonempty and unique")
    for path in (args.solver, args.stage_a, args.stage_b):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    stage_a_snapshot = args.report.parent / "stage_a_hu_average_policy.pt"
    stage_b_snapshot = args.report.parent / "stage_b_hu_average_policy.pt"
    print("Extracting lightweight HU AveragePolicy snapshots once in parent...", flush=True)
    stage_a_meta = _extract_hu_policy_snapshot(args.stage_a.resolve(), stage_a_snapshot)
    stage_b_meta = _extract_hu_policy_snapshot(args.stage_b.resolve(), stage_b_snapshot)
    print(
        "policy snapshots: "
        f"A={stage_a_meta['snapshot_bytes']/1048576.0:.2f} MiB "
        f"B={stage_b_meta['snapshot_bytes']/1048576.0:.2f} MiB",
        flush=True,
    )

    tasks: list[tuple[int, int, Episode, int]] = []
    hu_counts: dict[str, int] = {}
    total_counts: dict[str, int] = {}
    for seed in args.seeds:
        sampler = LegacyScenarioSampler(
            seed=int(seed) ^ 0x5CE0A710,
            config=LegacyScenarioConfig(),
        )
        hu = 0
        total = 0
        for index in range(int(args.scenarios_per_seed)):
            episode = sampler.sample_episode()
            deal_seed = _mix64(int(seed), int(index), 0xD34A1)
            total += 1
            if episode.game_is_hu:
                hu += 1
                tasks.append((int(seed), int(index), episode, int(deal_seed)))
        hu_counts[str(seed)] = int(hu)
        total_counts[str(seed)] = int(total)

    ctx = mp.get_context("spawn")
    rows: list[dict[str, Any]] = []
    workers = min(int(args.workers), max(1, len(tasks)))
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(
            str(args.solver.resolve()),
            str(stage_a_snapshot.resolve()),
            str(stage_b_snapshot.resolve()),
        ),
    ) as pool:
        for chunk in pool.map(_worker, tasks, chunksize=4):
            rows.extend(chunk)

    summary = _summarize(rows)

    report = {
        "schema": "SPINCORE_LT2_STAGE_A_B_FIRST_DIVERGENCE_FORENSIC_V1",
        "stage_a": {
            "path": str(args.stage_a.resolve()),
            "completed_iteration": int(stage_a_meta["completed_iteration"]),
            "worker_policy_snapshot_bytes": int(stage_a_meta["snapshot_bytes"]),
        },
        "stage_b": {
            "path": str(args.stage_b.resolve()),
            "completed_iteration": int(stage_b_meta["completed_iteration"]),
            "worker_policy_snapshot_bytes": int(stage_b_meta["snapshot_bytes"]),
        },
        "method": {
            "read_only": True,
            "new_training_roots": 0,
            "optimizer_steps": 0,
            "domain": "TRUE_HEADS_UP",
            "seeds": [int(x) for x in args.seeds],
            "scenarios_per_seed": int(args.scenarios_per_seed),
            "seed_family_role": (
                "already-seen diagnostic seeds from the powered weak-baseline gate; "
                "allowed for forensic localization only"
            ),
            "future_candidate_holdout_rule": (
                "any K4 or other intervention candidate must be accepted/rejected on "
                "a fresh seed family not used in this forensic design"
            ),
            "opponent_families": list(BASELINES),
            "deployed_policy_compared": "stored TRUE_HEADS_UP AveragePolicy",
            "worker_memory_strategy": (
                "full checkpoints are read once in the parent with mmap when available; "
                "workers load only derived HU AveragePolicy snapshots, never reservoirs"
            ),
            "workers": int(workers),
            "pairing": (
                "same HU scenario, solver deal, weak opponent, hero seat and random "
                "streams; states remain identical until sampled Stage-A/Stage-B hero "
                "actions first differ"
            ),
            "first_divergence_partition": list(GROUPS),
            "contribution_definition": (
                "mutually exclusive first-divergence group B-minus-A chip delta divided "
                "by seat-runs per scenario; group mean contributions sum to overall "
                "paired B-minus-A chip EV"
            ),
            "warning": (
                "localizes deployed-policy regression versus transparent weak baselines; "
                "does not prove a target-generation cause and is not exploitability/GTO"
            ),
        },
        "scenario_counts": {
            "total_by_seed": total_counts,
            "hu_by_seed": hu_counts,
            "hu_tasks": len(tasks),
            "seat_runs": len(rows),
        },
        "summary": summary,
    }

    for baseline in BASELINES:
        err = float(report["summary"][baseline]["contribution_closure_abs_error"])
        if not math.isfinite(err) or err > 1e-9:
            raise RuntimeError(f"contribution decomposition failed for {baseline}: {err}")

    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=== STAGE A -> STAGE B HU FIRST-DIVERGENCE FORENSIC ===")
    print(
        f"HU scenarios={len(tasks)} seat_runs={len(rows)} seeds={','.join(map(str,args.seeds))}"
    )
    for baseline in BASELINES:
        block = summary[baseline]
        overall = block["overall"]["b_minus_a_chip_ev"]
        print(
            f"{baseline}: B-A={overall['mean']:+.3f} "
            f"CI95=[{overall['ci95_low']:+.3f},{overall['ci95_high']:+.3f}] "
            f"divergence_rate={block['overall']['divergence_rate']:.3f}"
        )
        for group in GROUPS:
            g = block["first_divergence_groups"][group]
            c = g["contribution_to_overall_b_minus_a_chip_ev"]
            if int(g["seat_runs"]) > 0:
                print(
                    f"  {group}: freq={g['seat_run_fraction']:.3f} "
                    f"contrib={c['mean']:+.3f} "
                    f"CI95=[{c['ci95_low']:+.3f},{c['ci95_high']:+.3f}]"
                )
    print("LT2_STAGE_A_B_FIRST_DIVERGENCE_FORENSIC_COMPLETE")
    print(f"report={args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
