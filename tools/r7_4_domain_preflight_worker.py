from __future__ import annotations

import argparse
import json
from pathlib import Path

from spincore.solver import Episode, SolverLibrary


SCHEMA = "SPINCORE_R7_4_DOMAIN_PREFLIGHT_V1"
PAYOUT = (0.5, 0.3, 0.2)


def _finish_hand(root):
    state = root
    steps = 0
    while not state.terminal and steps < 64:
        legal = state.legal_actions()
        if not legal:
            raise RuntimeError("nonterminal state has no legal actions")
        action = 1 if 1 in legal else legal[0]  # CheckCall when legal.
        state.apply(action)
        steps += 1
    if not state.terminal:
        raise RuntimeError("structural preflight did not reach terminal within 64 abstract actions")
    chip = state.terminal_chip_delta()
    icm = state.terminal_icm_delta(PAYOUT)
    return {
        "terminal_steps": steps,
        "chip_delta": list(chip),
        "chip_zero_sum": sum(chip) == 0,
        "icm_delta": list(icm),
        "icm_zero_sum_within_1e12": abs(sum(icm)) <= 1e-12,
    }


def _case(solver: SolverLibrary, *, label: str, episode: Episode, seed: int, expected_domain: int):
    root = solver.create(episode, seed)
    try:
        if root.terminal:
            raise RuntimeError(f"{label}: root unexpectedly terminal")
        actor = root.actor
        legal = root.legal_actions()
        neural = root.neural_bytes()
        domain = root.domain
        if domain != expected_domain:
            raise RuntimeError(f"{label}: domain {domain} != expected {expected_domain}")
        if actor not in (0, 1, 2):
            raise RuntimeError(f"{label}: invalid actor {actor}")
        if not legal:
            raise RuntimeError(f"{label}: no legal actions")
        if not neural:
            raise RuntimeError(f"{label}: empty neural input")

        before = root.neural_bytes()
        clone = root.clone()
        try:
            if clone.neural_bytes() != before:
                raise RuntimeError(f"{label}: exact clone neural bytes differ")
        finally:
            clone.close()

        terminal = _finish_hand(root)
        if not terminal["chip_zero_sum"]:
            raise RuntimeError(f"{label}: terminal chip delta not zero-sum")
        if not terminal["icm_zero_sum_within_1e12"]:
            raise RuntimeError(f"{label}: terminal ICM delta not zero-sum within tolerance")
        return {
            "label": label,
            "seed": int(seed),
            "game_is_hu": bool(episode.game_is_hu),
            "dealer_id": int(episode.dealer_id),
            "stacks": list(episode.stacks),
            "dead_players": list(episode.dead_players),
            "domain": int(domain),
            "root_actor": int(actor),
            "root_legal_actions": list(legal),
            "neural_input_bytes": len(neural),
            "clone_neural_exact": True,
            **terminal,
        }
    finally:
        root.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Structural exact-engine preflight for R7.4 HU and three-handed domains")
    ap.add_argument("--solver", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    solver = SolverLibrary(args.solver)
    cases = []
    seeds = (17, 1234567, 0x715EED)

    # True HU with seat 0 dead; exercise both live dealer identities.
    for dealer in (1, 2):
        for seed in seeds:
            cases.append(_case(
                solver,
                label=f"HU_d{dealer}_s{seed}",
                episode=Episode(1500, True, 0, 10, 20, (0, 750, 750), dealer, (0,)),
                seed=seed,
                expected_domain=1,
            ))

    # Whole-hand three-handed domain; exercise all dealer rotations.
    for dealer in (0, 1, 2):
        for seed in seeds:
            cases.append(_case(
                solver,
                label=f"3H_d{dealer}_s{seed}",
                episode=Episode(1500, False, 0, 10, 20, (500, 500, 500), dealer, ()),
                seed=seed,
                expected_domain=0,
            ))

    hu = [row for row in cases if row["game_is_hu"]]
    three = [row for row in cases if not row["game_is_hu"]]
    payload = {
        "schema": SCHEMA,
        "solver": str(args.solver),
        "hu_case_count": len(hu),
        "three_handed_case_count": len(three),
        "case_count": len(cases),
        "hu_domains": sorted(set(row["domain"] for row in hu)),
        "three_handed_domains": sorted(set(row["domain"] for row in three)),
        "all_chip_zero_sum": all(row["chip_zero_sum"] for row in cases),
        "all_icm_zero_sum_within_1e12": all(row["icm_zero_sum_within_1e12"] for row in cases),
        "all_clone_neural_exact": all(row["clone_neural_exact"] for row in cases),
        "cases": cases,
        "strategic_gate_defined": False,
        "note": "Structural R7.4 preflight only. It proves both whole-hand domains can be instantiated/traversed under the exact solver source. It does not invent or satisfy a larger-pilot strategic/statistical acceptance gate.",
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
