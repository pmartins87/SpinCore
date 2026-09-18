#!/usr/bin/env python3
from __future__ import annotations

"""Cross-street future-chance variance audit on actual LT2 regression contexts.

This read-only diagnostic asks whether the HU-preflop board-noise mechanism is a
special Jammer artifact or a broader target-estimator problem.

It reconstructs three already-observed regression contexts from the forensic
seed family:
  * JAMMER preflop facing all-in, but with outcome-equivalent CHECK_CALL/ALL_IN
    collapsed into one CONTINUE class;
  * PASSIVE_CALLER first-divergence FLOP states;
  * UNIFORM_LEGAL first-divergence TURN states.

For each context and seed, it selects both FAILURE and CONTROL states.  Opponent
hole cards are held fixed at the actually dealt hidden hand.  Only unrevealed
future board cards are resampled.

Reference decomposition:
  * 8 independent future boards;
  * 4 target repeats per board;
  * exact_opponent_levels=1;
  * variance split into within-board opponent-action noise, future-board
    variance, and current-model error to the conditional mean.

Production-shaped candidates:
  * disjoint future-board stream;
  * exact_opponent_levels=0;
  * 8 boards;
  * K1 versus K4 future-board averaging.

No training memories are written and no optimizer step is run.
"""

import argparse
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import torch

import audit_lt2_hu_preflop_conditional_resampling as cond
import audit_lt2_hu_preflop_target_estimator_budget as base
import audit_lt2_jammer_facing_allin_target_overlay as jam
import audit_lt2_stage_a_b_first_divergence as fd
from audit_lt2_checkpoint_fit import _selftest_lean_rm_policy_tensor
from audit_lt2_repeated_target_variance import _capture_root_target
from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_solver_actions import apply_lean, lean_legal_actions
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.solver import Episode, SolverLibrary

FORENSIC_SEEDS = (20260920, 20260921, 20260922, 20260923, 20260924, 20260925)
GROUPS = (
    "JAMMER_PREFLOP_FACING_ALLIN_CLASS",
    "PASSIVE_CALLER_FLOP",
    "UNIFORM_LEGAL_TURN",
)
STATUS = ("FAILURE", "CONTROL")
BUDGETS = (1, 4)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver", type=Path, default=ROOT / "build/libspincore_solver_c.so")
    p.add_argument("--stage-a", type=Path, required=True)
    p.add_argument("--stage-b", type=Path, required=True)
    p.add_argument("--scenarios-per-seed", type=int, default=5000)
    p.add_argument("--states-per-seed-status", type=int, default=2)
    p.add_argument("--reference-boards", type=int, default=8)
    p.add_argument("--reference-repeats", type=int, default=4)
    p.add_argument("--candidate-boards", type=int, default=8)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--report", type=Path, required=True)
    return p.parse_args()


def _action_class(slot: int) -> str:
    if int(slot) == 0:
        return "FOLD"
    if int(slot) in (1, 9):
        return "CONTINUE"
    return f"OTHER_{int(slot)}"


def _target_group(baseline: str, state, last_actor: int | None, last_action: int | None, hero: int) -> str | None:
    street = fd._street(state)
    if baseline == "JAMMER":
        if (
            street == 0
            and last_actor is not None
            and int(last_actor) != int(hero)
            and int(last_action) == 9
        ):
            return "JAMMER_PREFLOP_FACING_ALLIN_CLASS"
        return None
    if baseline == "PASSIVE_CALLER" and street == 1:
        return "PASSIVE_CALLER_FLOP"
    if baseline == "UNIFORM_LEGAL" and street == 2:
        return "UNIFORM_LEGAL_TURN"
    return None


def _anchor_from_state(
    *,
    state,
    episode: Episode,
    seed: int,
    scenario_index: int,
    baseline: str,
    hero: int,
    action_path: list[int],
    a_slot: int,
    b_slot: int,
    status: str,
    group: str,
) -> dict[str, Any]:
    active_mask, legal = fd._legal_context(state)
    snapshot = state.deal_snapshot()
    live = [seat for seat, stack in enumerate(episode.stacks) if int(stack) > 0]
    if len(live) != 2:
        raise RuntimeError("cross-street anchor is not HU")
    opponent = live[1] if int(hero) == int(live[0]) else live[0]
    return {
        "seed": int(seed),
        "scenario_index": int(scenario_index),
        "baseline": str(baseline),
        "group": str(group),
        "status": str(status),
        "episode": episode,
        "actor": int(hero),
        "opponent_seat": int(opponent),
        "action_path": tuple(int(x) for x in action_path),
        "observation": state.neural_bytes(),
        "legal_mask": cond._legal_mask_tuple(tuple(legal)),
        "active_mask": int(active_mask),
        "a_slot": int(a_slot),
        "b_slot": int(b_slot),
        "a_class": _action_class(int(a_slot)),
        "b_class": _action_class(int(b_slot)),
        "holes": tuple(tuple(int(x) for x in row) for row in snapshot.holes),
        "board": tuple(int(x) for x in snapshot.board),
        "visible_board_count": int(snapshot.visible_board_count),
        "street": int(fd._street(state)),
    }


def _collect_seed_candidates(
    *,
    solver: SolverLibrary,
    stage_a: jam.StageModels,
    stage_b: jam.StageModels,
    seed: int,
    scenarios: int,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    out = {(g, s): [] for g in GROUPS for s in STATUS}
    sampler = LegacyScenarioSampler(
        seed=int(seed) ^ 0x5CE0A710,
        config=LegacyScenarioConfig(),
    )

    for scenario_index in range(int(scenarios)):
        episode = sampler.sample_episode()
        if not episode.game_is_hu:
            continue
        live = [seat for seat, stack in enumerate(episode.stacks) if int(stack) > 0]
        if len(live) != 2:
            raise RuntimeError("HU episode without exactly two live seats")
        deal_seed = fd._mix64(int(seed), int(scenario_index), 0xD34A1)

        for baseline in ("JAMMER", "PASSIVE_CALLER", "UNIFORM_LEGAL"):
            baseline_index = fd.BASELINES.index(baseline)
            for hero in live:
                state_a = solver.create(episode, int(deal_seed))
                state_b = solver.create(episode, int(deal_seed))
                rng_a = {
                    seat: random.Random(
                        fd._mix64(seed, scenario_index, seat, 100 + baseline_index)
                    )
                    for seat in range(3)
                }
                rng_b = {
                    seat: random.Random(
                        fd._mix64(seed, scenario_index, seat, 100 + baseline_index)
                    )
                    for seat in range(3)
                }
                rng_a[int(hero)] = random.Random(
                    fd._mix64(seed, scenario_index, hero, 777)
                )
                rng_b[int(hero)] = random.Random(
                    fd._mix64(seed, scenario_index, hero, 777)
                )
                action_path: list[int] = []
                last_actor = None
                last_action = None
                control_recorded: set[str] = set()
                try:
                    for _ in range(200):
                        if bool(state_a.terminal) != bool(state_b.terminal):
                            raise RuntimeError("paired terminal drift before selection")
                        if state_a.terminal:
                            break
                        actor = int(state_a.actor)
                        if actor != int(state_b.actor):
                            raise RuntimeError("paired actor drift before selection")

                        if actor != int(hero):
                            mask_a, slot_a = fd._baseline_action(
                                baseline, state_a, rng_a[actor]
                            )
                            mask_b, slot_b = fd._baseline_action(
                                baseline, state_b, rng_b[actor]
                            )
                            if mask_a != mask_b or slot_a != slot_b:
                                raise RuntimeError("baseline drift before selection")
                            apply_lean(state_a, mask_a, slot_a)
                            apply_lean(state_b, mask_b, slot_b)
                            action_path.append(int(slot_a))
                            last_actor, last_action = actor, int(slot_a)
                            continue

                        mask_a, legal_a, probs_a = stage_a.average_policy(state_a)
                        mask_b, legal_b, probs_b = stage_b.average_policy(state_b)
                        if mask_a != mask_b or legal_a != legal_b:
                            raise RuntimeError("A/B legal drift before selection")
                        slot_a = fd._sample_probs(tuple(legal_a), tuple(probs_a), rng_a[actor])
                        slot_b = fd._sample_probs(tuple(legal_b), tuple(probs_b), rng_b[actor])

                        group = _target_group(
                            baseline, state_a, last_actor, last_action, int(hero)
                        )

                        if group == "JAMMER_PREFLOP_FACING_ALLIN_CLASS":
                            class_a = _action_class(slot_a)
                            class_b = _action_class(slot_b)
                            if class_a != class_b:
                                out[(group, "FAILURE")].append(
                                    _anchor_from_state(
                                        state=state_a,
                                        episode=episode,
                                        seed=seed,
                                        scenario_index=scenario_index,
                                        baseline=baseline,
                                        hero=hero,
                                        action_path=action_path,
                                        a_slot=slot_a,
                                        b_slot=slot_b,
                                        status="FAILURE",
                                        group=group,
                                    )
                                )
                                break
                            if group not in control_recorded:
                                out[(group, "CONTROL")].append(
                                    _anchor_from_state(
                                        state=state_a,
                                        episode=episode,
                                        seed=seed,
                                        scenario_index=scenario_index,
                                        baseline=baseline,
                                        hero=hero,
                                        action_path=action_path,
                                        a_slot=slot_a,
                                        b_slot=slot_b,
                                        status="CONTROL",
                                        group=group,
                                    )
                                )
                                control_recorded.add(group)

                            # CHECK_CALL and ALL_IN are outcome-equivalent after an
                            # opponent jam.  To keep paired state identity if the raw
                            # slots differ inside CONTINUE, replay a canonical legal
                            # continuation in both arms.
                            if slot_a != slot_b and class_a == class_b == "CONTINUE":
                                canonical = 1 if 1 in legal_a else 9
                                apply_lean(state_a, mask_a, canonical)
                                apply_lean(state_b, mask_b, canonical)
                                action_path.append(int(canonical))
                            else:
                                apply_lean(state_a, mask_a, slot_a)
                                apply_lean(state_b, mask_b, slot_b)
                                action_path.append(int(slot_a))
                            last_actor, last_action = actor, int(action_path[-1])
                            continue

                        if int(slot_a) != int(slot_b):
                            if group in (
                                "PASSIVE_CALLER_FLOP",
                                "UNIFORM_LEGAL_TURN",
                            ):
                                out[(group, "FAILURE")].append(
                                    _anchor_from_state(
                                        state=state_a,
                                        episode=episode,
                                        seed=seed,
                                        scenario_index=scenario_index,
                                        baseline=baseline,
                                        hero=hero,
                                        action_path=action_path,
                                        a_slot=slot_a,
                                        b_slot=slot_b,
                                        status="FAILURE",
                                        group=group,
                                    )
                                )
                            break

                        if group in (
                            "PASSIVE_CALLER_FLOP",
                            "UNIFORM_LEGAL_TURN",
                        ) and group not in control_recorded:
                            out[(group, "CONTROL")].append(
                                _anchor_from_state(
                                    state=state_a,
                                    episode=episode,
                                    seed=seed,
                                    scenario_index=scenario_index,
                                    baseline=baseline,
                                    hero=hero,
                                    action_path=action_path,
                                    a_slot=slot_a,
                                    b_slot=slot_b,
                                    status="CONTROL",
                                    group=group,
                                )
                            )
                            control_recorded.add(group)

                        apply_lean(state_a, mask_a, slot_a)
                        apply_lean(state_b, mask_b, slot_b)
                        action_path.append(int(slot_a))
                        last_actor, last_action = actor, int(slot_a)
                    else:
                        raise RuntimeError("selection path exceeded 200 decisions")
                finally:
                    state_a.close()
                    state_b.close()

    return out


def _future_board(
    anchor: dict[str, Any],
    *,
    rng: random.Random,
) -> tuple[int, int, int, int, int]:
    holes = anchor["holes"]
    original = tuple(anchor["board"])
    visible = int(anchor["visible_board_count"])
    if visible not in (0, 3, 4):
        raise RuntimeError(f"unsupported live street visible board count {visible}")

    used = {
        int(card)
        for row in holes
        for card in row
        if int(card) >= 0
    }
    prefix = tuple(int(x) for x in original[:visible])
    used.update(prefix)
    remaining = [c for c in range(52) if c not in used]
    tail_count = 5 - visible
    tail = tuple(int(x) for x in rng.sample(remaining, tail_count))
    return prefix + tail


def _replay_anchor(
    *,
    solver: SolverLibrary,
    anchor: dict[str, Any],
    board: tuple[int, int, int, int, int],
):
    state = solver.create_with_deal(
        anchor["episode"],
        anchor["holes"],
        board,
    )
    try:
        for slot in anchor["action_path"]:
            if state.terminal:
                raise RuntimeError("generic replay became terminal before anchor")
            street = fd._street(state)
            active_mask = FIRST_RELEASE_ACTION_SPEC.active_mask(street)
            legal = lean_legal_actions(state, active_mask)
            if int(slot) not in legal:
                raise RuntimeError(
                    f"generic replay action {slot} illegal on street {street}"
                )
            apply_lean(state, active_mask, int(slot))

        if state.terminal:
            raise RuntimeError("generic replay anchor became terminal")
        if int(state.actor) != int(anchor["actor"]):
            raise RuntimeError("generic replay actor drift")
        if int(fd._street(state)) != int(anchor["street"]):
            raise RuntimeError("generic replay street drift")
        if state.neural_bytes() != anchor["observation"]:
            raise RuntimeError("generic replay observation drift")
        _, legal = fd._legal_context(state)
        if cond._legal_mask_tuple(tuple(legal)) != tuple(anchor["legal_mask"]):
            raise RuntimeError("generic replay legal-mask drift")
        return state
    except Exception:
        state.close()
        raise


def _capture(
    *,
    solver: SolverLibrary,
    runtime,
    anchor: dict[str, Any],
    board: tuple[int, int, int, int, int],
    completed: int,
    exact_level: int,
    rng_seed: int,
) -> tuple[torch.Tensor, int]:
    state = _replay_anchor(solver=solver, anchor=anchor, board=board)
    try:
        sample, nodes = _capture_root_target(
            runtime,
            state,
            iteration=int(completed) + 1,
            exact_level=int(exact_level),
            rng_seed=int(rng_seed),
        )
    finally:
        state.close()
    if sample.observation != anchor["observation"]:
        raise RuntimeError("cross-street target observation drift")
    if tuple(sample.legal) != tuple(anchor["legal_mask"]):
        raise RuntimeError("cross-street target legal drift")
    return torch.tensor(sample.target, dtype=torch.float32), int(nodes)


def _mse_over_legal(values: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    legal_f = legal.float()
    denom = legal_f.sum().clamp_min(1.0)
    return ((values * values) * legal_f).sum(dim=-1) / denom


def _candidate_block_metrics(
    *,
    targets: torch.Tensor,
    nodes: list[int],
    reference_target: torch.Tensor,
    legal: torch.Tensor,
    k: int,
) -> dict[str, Any]:
    if int(targets.shape[0]) % int(k) != 0:
        raise RuntimeError("candidate future-board pool not divisible by K")
    rows = []
    blocks = int(targets.shape[0]) // int(k)
    for b in range(blocks):
        lo = b * int(k)
        hi = lo + int(k)
        estimate = targets[lo:hi].mean(dim=0)
        m = base._policy_metrics(estimate, reference_target, legal)
        m["nodes"] = int(sum(nodes[lo:hi]))
        rows.append(m)
    fields = (
        "target_mse_to_reference",
        "policy_tv_to_reference",
        "candidate_policy_regret_to_reference_best_action_chips",
        "nodes",
    )
    out = {
        field: float(statistics.fmean(float(r[field]) for r in rows))
        for field in fields
    }
    out["argmax_agreement_rate"] = float(
        statistics.fmean(1.0 if r["argmax_agreement"] else 0.0 for r in rows)
    )
    return out


def _anchor_metrics(
    *,
    solver: SolverLibrary,
    runtime,
    anchor: dict[str, Any],
    completed: int,
    reference_boards: int,
    reference_repeats: int,
    candidate_boards: int,
) -> dict[str, Any]:
    legal = torch.tensor(anchor["legal_mask"], dtype=torch.bool)
    if int(legal.sum().item()) <= 0:
        raise RuntimeError("anchor legal mask empty")

    ref_targets = []
    ref_nodes = []
    for bpos in range(int(reference_boards)):
        board_rng = random.Random(
            fd._mix64(
                anchor["seed"],
                anchor["scenario_index"],
                anchor["anchor_index"],
                bpos,
                0xC4A9CE,
            )
        )
        board = _future_board(anchor, rng=board_rng)
        reps = []
        rep_nodes = []
        for rep in range(int(reference_repeats)):
            target, nodes = _capture(
                solver=solver,
                runtime=runtime,
                anchor=anchor,
                board=board,
                completed=completed,
                exact_level=1,
                rng_seed=fd._mix64(
                    anchor["seed"],
                    anchor["scenario_index"],
                    anchor["anchor_index"],
                    bpos,
                    rep,
                    0xE1AC7,
                ),
            )
            reps.append(target)
            rep_nodes.append(int(nodes))
        ref_targets.append(torch.stack(reps))
        ref_nodes.append(rep_nodes)

    ref = torch.stack(ref_targets)  # [boards, repeats, actions]
    board_mean = ref.mean(dim=1)
    grand_mean = board_mean.mean(dim=0)
    within_action = float(
        _mse_over_legal(ref - board_mean.unsqueeze(1), legal).mean().item()
    )
    future_board = float(
        _mse_over_legal(board_mean - grand_mean.unsqueeze(0), legal).mean().item()
    )

    raw_model = cond._model_raw(
        runtime, anchor["observation"], tuple(anchor["legal_mask"])
    )
    model_error = float(_mse_over_legal(raw_model - grand_mean, legal).item())
    sample_target_mse = float(
        _mse_over_legal(ref - raw_model, legal).mean().item()
    )
    component_sum = within_action + future_board + model_error

    cand_targets = []
    cand_nodes = []
    for bpos in range(int(candidate_boards)):
        board_rng = random.Random(
            fd._mix64(
                anchor["seed"],
                anchor["scenario_index"],
                anchor["anchor_index"],
                bpos,
                0xCAAD1D,
            )
        )
        board = _future_board(anchor, rng=board_rng)
        target, nodes = _capture(
            solver=solver,
            runtime=runtime,
            anchor=anchor,
            board=board,
            completed=completed,
            exact_level=0,
            rng_seed=fd._mix64(
                anchor["seed"],
                anchor["scenario_index"],
                anchor["anchor_index"],
                bpos,
                0xE0CAAD,
            ),
        )
        cand_targets.append(target)
        cand_nodes.append(int(nodes))
    cand = torch.stack(cand_targets)

    candidate = {
        str(k): _candidate_block_metrics(
            targets=cand,
            nodes=cand_nodes,
            reference_target=grand_mean,
            legal=legal,
            k=int(k),
        )
        for k in BUDGETS
    }

    return {
        "group": anchor["group"],
        "status": anchor["status"],
        "seed": int(anchor["seed"]),
        "scenario_index": int(anchor["scenario_index"]),
        "street": int(anchor["street"]),
        "visible_board_count": int(anchor["visible_board_count"]),
        "a_slot": int(anchor["a_slot"]),
        "b_slot": int(anchor["b_slot"]),
        "a_class": anchor["a_class"],
        "b_class": anchor["b_class"],
        "legal_action_count": int(legal.sum().item()),
        "variance": {
            "within_board_opponent_action": within_action,
            "future_board": future_board,
            "model_to_board_conditional_mean": model_error,
            "sample_target_mse": sample_target_mse,
            "component_sum": component_sum,
            "closure_abs_error": float(abs(component_sum - sample_target_mse)),
            "future_board_fraction_of_sample_mse": float(
                future_board / sample_target_mse
            ) if sample_target_mse > 0 else 0.0,
            "within_action_fraction_of_sample_mse": float(
                within_action / sample_target_mse
            ) if sample_target_mse > 0 else 0.0,
            "model_fraction_of_sample_mse": float(
                model_error / sample_target_mse
            ) if sample_target_mse > 0 else 0.0,
            "reference_mean_nodes_per_capture": float(
                statistics.fmean(
                    float(x) for row in ref_nodes for x in row
                )
            ),
        },
        "candidate": candidate,
    }


def _mean_ci(values: list[float]) -> dict[str, float | int]:
    return base._mean_ci([float(x) for x in values])


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "within_board_opponent_action",
        "future_board",
        "model_to_board_conditional_mean",
        "sample_target_mse",
        "future_board_fraction_of_sample_mse",
        "within_action_fraction_of_sample_mse",
        "model_fraction_of_sample_mse",
        "reference_mean_nodes_per_capture",
    )
    out = {
        "n": len(rows),
        "variance": {
            field: _mean_ci([r["variance"][field] for r in rows])
            for field in fields
        },
        "candidate": {},
    }
    for k in BUDGETS:
        out["candidate"][str(k)] = {
            field: _mean_ci(
                [r["candidate"][str(k)][field] for r in rows]
            )
            for field in (
                "target_mse_to_reference",
                "policy_tv_to_reference",
                "candidate_policy_regret_to_reference_best_action_chips",
                "nodes",
                "argmax_agreement_rate",
            )
        }
    out["k4_minus_k1"] = {
        field: _mean_ci(
            [
                r["candidate"]["4"][field] - r["candidate"]["1"][field]
                for r in rows
            ]
        )
        for field in (
            "target_mse_to_reference",
            "policy_tv_to_reference",
            "candidate_policy_regret_to_reference_best_action_chips",
            "nodes",
            "argmax_agreement_rate",
        )
    }
    return out


def main() -> int:
    args = parse_args()
    if args.scenarios_per_seed <= 0 or args.states_per_seed_status <= 0:
        raise SystemExit("positive scenario/state counts required")
    if args.reference_boards < 2 or args.reference_repeats < 2:
        raise SystemExit("reference requires >=2 boards and >=2 repeats")
    if args.candidate_boards < 4 or args.candidate_boards % 4 != 0:
        raise SystemExit("candidate boards must be >=4 and divisible by 4")
    if args.threads <= 0:
        raise SystemExit("positive thread count required")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(args.threads))
    _selftest_lean_rm_policy_tensor()
    solver = SolverLibrary(args.solver.resolve(strict=True))
    if not solver.explicit_deal_available:
        raise SystemExit("explicit-deal solver API required")

    stage_a_snapshot = args.report.parent / "stage_a_hu_models.pt"
    stage_b_snapshot = args.report.parent / "stage_b_hu_models.pt"
    meta_a = jam._extract_stage_snapshot(
        args.stage_a.resolve(strict=True), stage_a_snapshot
    )
    meta_b = jam._extract_stage_snapshot(
        args.stage_b.resolve(strict=True), stage_b_snapshot
    )
    stage_a = jam.StageModels(stage_a_snapshot, solver)
    stage_b = jam.StageModels(stage_b_snapshot, solver)

    selected: list[dict[str, Any]] = []
    candidate_counts: dict[str, Any] = {}
    for seed in FORENSIC_SEEDS:
        pools = _collect_seed_candidates(
            solver=solver,
            stage_a=stage_a,
            stage_b=stage_b,
            seed=int(seed),
            scenarios=int(args.scenarios_per_seed),
        )
        candidate_counts[str(seed)] = {}
        for group in GROUPS:
            candidate_counts[str(seed)][group] = {}
            for status in STATUS:
                pool = pools[(group, status)]
                candidate_counts[str(seed)][group][status] = len(pool)
                need = int(args.states_per_seed_status)
                if len(pool) < need:
                    raise RuntimeError(
                        f"seed={seed} group={group} status={status} "
                        f"only {len(pool)} candidates; need {need}"
                    )
                chooser = random.Random(
                    fd._mix64(
                        seed,
                        GROUPS.index(group),
                        STATUS.index(status),
                        0x5E1EC7,
                    )
                )
                indices = sorted(chooser.sample(range(len(pool)), need))
                for idx in indices:
                    selected.append(pool[idx])

    for idx, anchor in enumerate(selected):
        anchor["anchor_index"] = int(idx)

    rows = []
    for idx, anchor in enumerate(selected):
        print(
            f"CROSS_STREET anchor={idx+1}/{len(selected)} "
            f"group={anchor['group']} status={anchor['status']} "
            f"seed={anchor['seed']} scenario={anchor['scenario_index']} "
            f"transition={anchor['a_slot']}->{anchor['b_slot']}",
            flush=True,
        )
        row = _anchor_metrics(
            solver=solver,
            runtime=stage_b.runtime,
            anchor=anchor,
            completed=int(meta_b["completed_iteration"]),
            reference_boards=int(args.reference_boards),
            reference_repeats=int(args.reference_repeats),
            candidate_boards=int(args.candidate_boards),
        )
        if row["variance"]["closure_abs_error"] > 1e-5:
            raise RuntimeError(
                "variance decomposition closure failed: "
                f"{row['variance']['closure_abs_error']}"
            )
        rows.append(row)

    summaries = {}
    for group in GROUPS:
        summaries[group] = {}
        for status in STATUS:
            rr = [
                r for r in rows
                if r["group"] == group and r["status"] == status
            ]
            summaries[group][status] = _aggregate(rr)

        failure = summaries[group]["FAILURE"]
        control = summaries[group]["CONTROL"]
        summaries[group]["FAILURE_MINUS_CONTROL"] = {
            "future_board_fraction_of_sample_mse": _mean_ci(
                [
                    r["variance"]["future_board_fraction_of_sample_mse"]
                    for r in rows
                    if r["group"] == group and r["status"] == "FAILURE"
                ]
            ),
            "note": (
                "failure/control CIs are reported separately above; this block does "
                "not pretend the independently sampled anchors are paired"
            ),
        }

    report = {
        "schema": "SPINCORE_LT2_CROSS_STREET_FUTURE_CHANCE_AUDIT_V1",
        "stage_a": {
            "checkpoint": str(args.stage_a.resolve()),
            "completed_iteration": int(meta_a["completed_iteration"]),
        },
        "stage_b": {
            "checkpoint": str(args.stage_b.resolve()),
            "completed_iteration": int(meta_b["completed_iteration"]),
        },
        "method": {
            "read_only": True,
            "training_memory_writes": 0,
            "optimizer_steps": 0,
            "new_training_roots": 0,
            "forensic_seeds": list(FORENSIC_SEEDS),
            "future_holdout_seeds_touched": False,
            "groups": list(GROUPS),
            "status": list(STATUS),
            "states_per_seed_status": int(args.states_per_seed_status),
            "selection": {
                "jammer": (
                    "preflop after opponent ALL_IN; CHECK_CALL and ALL_IN collapsed "
                    "to the benchmark-equivalent CONTINUE class; FAILURE means "
                    "Stage A/B sampled different FOLD-vs-CONTINUE classes"
                ),
                "passive": "first A/B raw-action divergence on FLOP",
                "uniform": "first A/B raw-action divergence on TURN",
                "control": (
                    "same context reached before any non-equivalent A/B divergence, "
                    "with the compared action/class agreeing"
                ),
            },
            "hidden_hand": "actual dealt opponent hand held fixed",
            "future_chance": (
                "already-visible board prefix held fixed; only unrevealed future "
                "board cards resampled"
            ),
            "reference": {
                "future_boards": int(args.reference_boards),
                "target_repeats_per_board": int(args.reference_repeats),
                "exact_opponent_levels": 1,
            },
            "candidate": {
                "future_boards": int(args.candidate_boards),
                "exact_opponent_levels": 0,
                "budgets": list(BUDGETS),
                "sampling_stream_disjoint_from_reference": True,
            },
            "warning": (
                "conditional on actual hidden hand; isolates future-chance variance "
                "rather than full information-set hidden-hand variance"
            ),
        },
        "candidate_counts_by_seed": candidate_counts,
        "summaries": summaries,
        "rows": rows,
    }
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== CROSS-STREET FUTURE-CHANCE AUDIT ===")
    for group in GROUPS:
        print(group)
        for status in STATUS:
            s = summaries[group][status]
            v = s["variance"]
            k = s["k4_minus_k1"]
            print(
                f"  {status}: n={s['n']} "
                f"board_fraction={v['future_board_fraction_of_sample_mse']['mean']:.3f} "
                f"action_fraction={v['within_action_fraction_of_sample_mse']['mean']:.3f} "
                f"model_fraction={v['model_fraction_of_sample_mse']['mean']:.3f} "
                f"K4-K1_mse={k['target_mse_to_reference']['mean']:+.6f} "
                f"K4-K1_regret={k['candidate_policy_regret_to_reference_best_action_chips']['mean']:+.2f}"
            )
    print("LT2_CROSS_STREET_FUTURE_CHANCE_AUDIT_COMPLETE")
    print(f"report={args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
