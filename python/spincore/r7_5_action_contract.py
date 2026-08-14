from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

SLOT_BY_NAME = {
    "FOLD": 0,
    "CHECK_CALL": 1,
    "MIN_RAISE": 2,
    "POT_33": 3,
    "POT_40": 4,
    "POT_50": 5,
    "POT_66": 6,
    "POT_75": 7,
    "POT_100": 8,
    "ALL_IN": 9,
}
NAME_BY_SLOT = {value: key for key, value in SLOT_BY_NAME.items()}
UNIVERSAL_ACTION_COUNT = 10
NON_AGGRESSIVE = ("FOLD", "CHECK_CALL")

V1_SCHEMA = "SPINCORE_R7_5_4_ACTION_ABSTRACTION_ABLATION_PRECOMMIT_V1"
V2_SCHEMA = "SPINCORE_R7_5_4_ACTION_ABSTRACTION_ABLATION_PRECOMMIT_V2"
V3_SCHEMA = "SPINCORE_R7_5_4_ACTION_ABSTRACTION_ABLATION_PRECOMMIT_V3"


@dataclass(frozen=True)
class ActionCandidateSpec:
    candidate_id: str
    preflop_mask: int
    postflop_mask: int
    eligible_to_win: bool
    phase: str

    def active_mask(self, street: int) -> int:
        return self.preflop_mask if int(street) == 0 else self.postflop_mask


def universal_mask(names) -> int:
    mask = 0
    for name in names:
        try:
            slot = SLOT_BY_NAME[str(name)]
        except KeyError as exc:
            raise ValueError(f"unknown universal action name: {name!r}") from exc
        mask |= 1 << slot
    if mask & ~0x3FF:
        raise ValueError("universal action mask overflow")
    return mask


def mask_with_passive(aggressive_names) -> int:
    return universal_mask((*NON_AGGRESSIVE, *(str(x) for x in aggressive_names)))


def mask_names(mask: int) -> tuple[str, ...]:
    value = int(mask)
    if value < 0 or value > 0x3FF:
        raise ValueError("invalid universal action mask")
    return tuple(NAME_BY_SLOT[slot] for slot in range(UNIVERSAL_ACTION_COUNT) if value & (1 << slot))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_authoritative_action_contract(repo_root: str | Path) -> tuple[dict, dict, dict]:
    root = Path(repo_root)
    base = root / "validation"
    v1 = _load_json(base / "R7_5_4_ACTION_ABSTRACTION_ABLATION_PRECOMMIT.json")
    v2 = _load_json(base / "R7_5_4_ACTION_ABSTRACTION_ABLATION_PRECOMMIT_V2.json")
    v3 = _load_json(base / "R7_5_4_ACTION_ABSTRACTION_ABLATION_PRECOMMIT_V3.json")
    if v1.get("schema") != V1_SCHEMA or v2.get("schema") != V2_SCHEMA or v3.get("schema") != V3_SCHEMA:
        raise ValueError("R7.5.4 authoritative precommit chain schema mismatch")
    if v1.get("universal_output_width") != UNIVERSAL_ACTION_COUNT:
        raise ValueError("R7.5.4 universal action width drifted")
    if list(v1.get("universal_action_vocabulary") or []) != [NAME_BY_SLOT[i] for i in range(10)]:
        raise ValueError("R7.5.4 universal action vocabulary drifted")
    if not v2["corrections"]["PF0_CONTROL_33_75_AI_exact_legacy_equivalence_required"]:
        raise ValueError("R7.5.4 llround/control-equivalence correction missing")
    if v3["seed_derivation"]["paired_evaluation_seeds"] != [1817694185, 1617273629]:
        raise ValueError("R7.5.4 authoritative evaluation seed drift")
    if any(bool(doc.get("ready_for_tables")) for doc in (v1, v2, v3)):
        raise ValueError("R7.5.4 precommit chain illegally authorizes table use")
    return v1, v2, v3


def postflop_candidate_specs(repo_root: str | Path) -> dict[str, ActionCandidateSpec]:
    v1, _, _ = load_authoritative_action_contract(repo_root)
    phase = v1["subgates"]["R7_5_4A_POSTFLOP"]
    preflop = mask_with_passive(phase["preflop_fixed"])
    out = {}
    for row in phase["candidates"]:
        candidate_id = str(row["id"])
        out[candidate_id] = ActionCandidateSpec(
            candidate_id=candidate_id,
            preflop_mask=preflop,
            postflop_mask=mask_with_passive(row["postflop"]),
            eligible_to_win=bool(row["eligible_to_win"]),
            phase="R7_5_4A_POSTFLOP",
        )
    return out


def preflop_candidate_specs(
    repo_root: str | Path,
    *,
    selected_postflop_candidate: str,
) -> dict[str, ActionCandidateSpec]:
    v1, _, _ = load_authoritative_action_contract(repo_root)
    postflop = postflop_candidate_specs(repo_root)
    if selected_postflop_candidate not in postflop:
        raise ValueError("unknown selected R7.5.4A postflop candidate")
    fixed_postflop_mask = postflop[selected_postflop_candidate].postflop_mask
    phase = v1["subgates"]["R7_5_4B_PREFLOP"]
    out = {}
    for row in phase["candidates"]:
        candidate_id = str(row["id"])
        out[candidate_id] = ActionCandidateSpec(
            candidate_id=candidate_id,
            preflop_mask=mask_with_passive(row["preflop"]),
            postflop_mask=fixed_postflop_mask,
            eligible_to_win=bool(row["eligible_to_win"]),
            phase="R7_5_4B_PREFLOP",
        )
    return out
