#!/usr/bin/env python3
from __future__ import annotations

"""Freeze the three source-valid DC0 action-sizing sentinels for real OpenHoldem.

This packet is intentionally NOT a search.  It regenerates three deterministic
THREE_HANDED / PASSIVE cases discovered and validated by the DC0 semantic search:
- TECH_BETPOT_075: attempt 1 / deal seed 20260924 / stack 500
- ALLIN_ADJUSTED: attempt 8 / deal seed 20260931 / stack 160
- TECH_BETPOT_033: attempt 72 / deal seed 20260995 / stack 500

Each case is rendered as a standalone TestSuite2 file.  The packet manifest
records the exact target decision(s), and generation fails if the expected
family is no longer present.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(ROOT / "python"))

import generate_deepcrusher_openholdem_testsuite as base  # noqa: E402
from spincore.deepcrusher_policy import DeepCrusherR8Policy  # noqa: E402
from spincore.solver import SolverLibrary  # noqa: E402


SPECS = (
    {
        "id": "dc0_real_075",
        "family": "TECH_BETPOT_075",
        "domain": "THREE_HANDED",
        "attempt": 1,
        "deal_seed": 20260924,
        "stack_chips": 500,
        "opponent_style": "PASSIVE",
    },
    {
        "id": "dc0_real_allin_adjusted",
        "family": "ALLIN_ADJUSTED",
        "domain": "THREE_HANDED",
        "attempt": 8,
        "deal_seed": 20260931,
        "stack_chips": 160,
        "opponent_style": "PASSIVE",
    },
    {
        "id": "dc0_real_033",
        "family": "TECH_BETPOT_033",
        "domain": "THREE_HANDED",
        "attempt": 72,
        "deal_seed": 20260995,
        "stack_chips": 500,
        "opponent_style": "PASSIVE",
    },
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solver", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    solver = SolverLibrary(args.solver)
    if not solver.explicit_deal_available:
        raise SystemExit("real parity packet requires explicit deal snapshot ABI")
    policy = DeepCrusherR8Policy.from_repository(ROOT)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema": "SPINCORE_DEEPCRUSHER_REAL_OPENHOLDEM_PARITY_V1",
        "status": "FROZEN_OFFLINE_EXPECTATIONS_PENDING_REAL_OPENHOLDEM",
        "canonical_parity_pass": False,
        "required_next_system": "REAL_OPENHOLDEM_MANUALMODE_TESTSUITE2",
        "source_generator_seed": 20260923,
        "cases": [],
    }

    for spec in SPECS:
        attempt = int(spec["attempt"])
        dealer = attempt % 3
        # Same deterministic seat-cycling contract as the semantic search.
        hero_seat = (attempt // (3 * 8)) % 3
        episode = base.make_episode(
            str(spec["domain"]),
            dealer,
            stack_chips=int(spec["stack_chips"]),
        )
        deal, actions, _ = base.play_candidate(
            solver,
            policy,
            episode,
            hero_seat=hero_seat,
            deal_seed=int(spec["deal_seed"]),
            case_index=attempt,
            opponent_style=str(spec["opponent_style"]),
        )
        hero_rows = [row for row in actions if row.hero]
        targets = [row for row in hero_rows if row.decision_family == spec["family"]]
        if not targets:
            found = sorted({str(row.decision_family) for row in hero_rows})
            raise RuntimeError(
                f"{spec['id']}: expected {spec['family']} not present; found={found}"
            )

        case_text = base.render_case(
            case_id=str(spec["id"]),
            episode=episode,
            deal=deal,
            actions=actions,
            hero_seat=hero_seat,
        )
        case_path = args.output_dir / f"{spec['id']}.txt"
        case_path.write_text(case_text, encoding="utf-8")

        target_rows = [
            {
                "street": int(row.street),
                "action": base.ACTION_TEXT[int(row.action_type)],
                "amount_to": int(row.amount_to),
                "legal_buttons": str(row.legal_buttons),
                "decision_family": str(row.decision_family),
                "hand_class": (
                    None
                    if row.decision_metadata is None
                    else row.decision_metadata.get("hand_class")
                ),
                "openppl_main": (
                    None
                    if row.decision_metadata is None
                    else row.decision_metadata.get("openppl_main")
                ),
                "allin_adjustment_ratio": (
                    None
                    if row.decision_metadata is None
                    else row.decision_metadata.get("allin_adjustment_ratio")
                ),
            }
            for row in targets
        ]
        manifest["cases"].append(
            {
                **spec,
                "dealer": int(dealer),
                "hero_seat": int(hero_seat),
                "file": case_path.name,
                "target_rows": target_rows,
            }
        )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    collection_path = args.output_dir / "deepcrusher_dc0_real_parity.tc"
    collection_path.write_text(
        "".join(f"{case['file']}\n" for case in manifest["cases"]),
        encoding="utf-8",
    )

    print(manifest_path.read_text(encoding="utf-8"), end="")
    print(collection_path.read_text(encoding="utf-8"), end="")
    for case in manifest["cases"]:
        print(f"=== REAL_PARITY_CASE_BEGIN:{case['file']} ===")
        print((args.output_dir / case["file"]).read_text(encoding="utf-8"), end="")
        print(f"=== REAL_PARITY_CASE_END:{case['file']} ===")
    print("DEEPC_RUSHER_REAL_OPENHOLDEM_PARITY_PACKET_GENERATION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
