from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from spincore.solver import Episode, SolverLibrary
from spincore_nn.codec import decode_spnniv1
from spincore_nn.codec_v2 import decode_spnniv2

SCHEMA = "SPINCORE_R7_5_REAL_GAME_REACHABILITY_AUDIT_V1"
FULL_UNIVERSAL_MASK = 0x3FF
SEEDS = (1, 7, 31, 127, 509, 20260814)
PAYOUT = (0.5, 0.3, 0.2)


def episodes() -> tuple[tuple[str, Episode], ...]:
    return (
        ("HU_DEAD_SEAT_0", Episode(1500, True, 0, 10, 20, (0, 750, 750), 1, (0,))),
        ("HU_DEAD_SEAT_1", Episode(1500, True, 0, 10, 20, (750, 0, 750), 2, (1,))),
        ("3H_EQUAL", Episode(1500, False, 0, 10, 20, (500, 500, 500), 0, ())),
        ("3H_ASCENDING", Episode(1500, False, 0, 10, 20, (250, 500, 750), 1, ())),
        ("3H_DESCENDING", Episode(1500, False, 0, 10, 20, (700, 500, 300), 2, ())),
    )


def _expected_legacy_mask(legal: tuple[int, ...]) -> tuple[int, ...]:
    legal_set = set(int(action) for action in legal)
    return tuple(1 if action in legal_set else 0 for action in range(6))


def _walk(solver: SolverLibrary, episode: Episode, *, seed: int, universal: bool) -> dict:
    state = solver.create(episode, int(seed))
    metrics = {
        "decisions": 0,
        "v1_legal_mask_checks": 0,
        "v2_legal_mask_checks": 0,
        "legacy_illegal_rejections": 0,
        "universal_illegal_rejections": 0,
        "history_monotonic_checks": 0,
        "terminal_chip_conservation_checks": 0,
        "terminal_icm_conservation_checks": 0,
    }
    failures: list[str] = []
    previous_history_len = 0
    try:
        for depth in range(128):
            if state.terminal:
                if state.legal_actions() != ():
                    failures.append("terminal_state_exposed_legacy_legal_action")
                chip_delta = state.terminal_chip_delta()
                metrics["terminal_chip_conservation_checks"] += 1
                if sum(chip_delta) != 0:
                    failures.append(f"terminal_chip_delta_not_zero_sum:{chip_delta}")
                icm_delta = state.terminal_icm_delta(PAYOUT)
                metrics["terminal_icm_conservation_checks"] += 1
                if abs(sum(icm_delta)) > 1e-10:
                    failures.append(f"terminal_icm_delta_not_zero_sum:{icm_delta}")
                break

            metrics["decisions"] += 1
            if state.actor not in (0, 1, 2):
                failures.append(f"invalid_actor:{state.actor}")
                break

            legal = state.legal_actions()
            if not legal:
                failures.append("nonterminal_without_legacy_legal_action")
                break
            expected = _expected_legacy_mask(legal)
            v1 = decode_spnniv1(state.neural_bytes())
            v2 = decode_spnniv2(state.neural_bytes_v2())
            metrics["v1_legal_mask_checks"] += 1
            metrics["v2_legal_mask_checks"] += 1
            if tuple(int(x) for x in v1.legal) != expected:
                failures.append("spnniv1_legal_mask_diverged_from_engine")
                break
            if tuple(int(x) for x in v2.legal) != expected:
                failures.append("spnniv2_legal_mask_diverged_from_engine")
                break
            if not (0 <= int(v1.history_len) <= 32 and 0 <= int(v2.history_len) <= 32):
                failures.append("history_length_out_of_wire_bounds")
                break
            metrics["history_monotonic_checks"] += 1
            if int(v2.history_len) < previous_history_len:
                failures.append(
                    f"public_history_moved_backwards:{previous_history_len}->{v2.history_len}"
                )
                break
            previous_history_len = int(v2.history_len)

            illegal_legacy = next((action for action in range(6) if action not in set(legal)), None)
            if illegal_legacy is not None:
                clone = state.clone()
                try:
                    try:
                        clone.apply(int(illegal_legacy))
                    except RuntimeError:
                        metrics["legacy_illegal_rejections"] += 1
                    else:
                        failures.append(f"legacy_illegal_action_accepted:{illegal_legacy}")
                        break
                finally:
                    clone.close()

            universal_legal = state.universal_legal_actions(FULL_UNIVERSAL_MASK)
            if not universal_legal:
                failures.append("nonterminal_without_universal_legal_action")
                break
            illegal_universal = next(
                (action for action in range(10) if action not in set(universal_legal)), None
            )
            if illegal_universal is not None:
                clone = state.clone()
                try:
                    try:
                        clone.apply_universal(FULL_UNIVERSAL_MASK, int(illegal_universal))
                    except RuntimeError:
                        metrics["universal_illegal_rejections"] += 1
                    else:
                        failures.append(f"universal_illegal_action_accepted:{illegal_universal}")
                        break
                finally:
                    clone.close()

            if universal:
                action = universal_legal[(int(seed) * 3 + depth) % len(universal_legal)]
                state.apply_universal(FULL_UNIVERSAL_MASK, int(action))
            else:
                action = legal[(int(seed) + depth) % len(legal)]
                state.apply(int(action))
        else:
            failures.append("trajectory_exceeded_128_decisions")

        if not state.terminal and not failures:
            failures.append("trajectory_ended_without_terminal_state")
    finally:
        state.close()

    return {"pass": not failures, "failures": failures, **metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit legally reachable HU/3H SpinCore trajectories")
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execution-sha", required=True)
    args = parser.parse_args()

    started = time.time()
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))
    solver = SolverLibrary(args.lib)
    rows = []
    totals = {
        "trajectories": 0,
        "decisions": 0,
        "v1_legal_mask_checks": 0,
        "v2_legal_mask_checks": 0,
        "legacy_illegal_rejections": 0,
        "universal_illegal_rejections": 0,
        "history_monotonic_checks": 0,
        "terminal_chip_conservation_checks": 0,
        "terminal_icm_conservation_checks": 0,
    }
    for episode_name, episode in episodes():
        for seed in SEEDS:
            for mode in ("LEGACY", "UNIVERSAL"):
                result = _walk(solver, episode, seed=int(seed), universal=(mode == "UNIVERSAL"))
                row = {
                    "episode": episode_name,
                    "seed": int(seed),
                    "mode": mode,
                    **result,
                }
                rows.append(row)
                totals["trajectories"] += 1
                for key in totals:
                    if key != "trajectories":
                        totals[key] += int(result[key])

    passed = all(bool(row["pass"]) for row in rows)
    payload = {
        "schema": SCHEMA,
        "execution_sha": str(args.execution_sha),
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "episode_count": len(episodes()),
        "seeds": list(SEEDS),
        "modes": ["LEGACY", "UNIVERSAL"],
        "rows": rows,
        "totals": totals,
        "reachability_gate_pass": bool(passed),
        "claims": {
            "observations_only_from_engine_reached_states": True,
            "spnniv1_legal_mask_matches_engine_on_audited_states": bool(passed),
            "spnniv2_legal_mask_matches_engine_on_audited_states": bool(passed),
            "illegal_actions_fail_closed_on_audited_states": bool(passed),
            "terminal_chip_and_icm_conservation_on_audited_states": bool(passed),
            "universal_action_path_traversed": True,
            "legacy_action_path_traversed": True,
        },
        "strategic_output": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
