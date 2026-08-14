from __future__ import annotations

import pytest

from spincore.r7_5_action_aggregate import (
    CandidateSelectionEvidence,
    DomainSelectionEvidence,
    prune_action_level,
)

DOMAINS = ("TRUE_HEADS_UP", "THREE_HANDED")
DENSE = "PF_DENSE_REFERENCE"
PF0 = "PF0_CONTROL_33_75_AI"
PF1 = "PF1_33_50_75_AI"
PF2 = "PF2_33_50_75_100_AI"


def _domain(omission: float, crossplay: float, nodes=100.0, seconds=2.0, branches=3.0, n=64):
    return DomainSelectionEvidence(
        omission_samples=(float(omission),) * n,
        crossplay_samples=(float(crossplay),) * n,
        nodes_per_root=float(nodes),
        tree_seconds_per_root=float(seconds),
        effective_branches_per_decision=float(branches),
    )


def _candidate(cid: str, *, omission=0.0, crossplay=0.0, nodes=100.0, seconds=2.0, branches=3.0, learning=True):
    return CandidateSelectionEvidence(
        candidate_id=cid,
        learning_gate_pass=bool(learning),
        domains={
            domain: _domain(omission, crossplay, nodes, seconds, branches)
            for domain in DOMAINS
        },
    )


def _dense(learning=True):
    return _candidate(DENSE, omission=0.0, crossplay=0.0, learning=learning)


def test_160_prunes_material_omission_loss_but_never_final_selects() -> None:
    evidence = {
        DENSE: _dense(),
        PF0: _candidate(PF0, omission=0.001),
        PF1: _candidate(PF1, omission=0.0015),  # difference .0005 < material floor
        PF2: _candidate(PF2, omission=0.020),   # clearly material
    }
    out = prune_action_level(evidence, root_level=160)
    assert out["status"] == "PASS_LEVEL"
    assert PF2 not in out["survivors"]
    assert set(out["survivors"]) == {PF0, PF1}
    assert out["selected_candidate"] is None
    assert out["next_level"] == 320
    assert set(out["mandatory_next_level_execution"]) == {PF0, PF1, DENSE}
    assert out["fallback_used"] is False


def test_160_does_not_use_rank7_to_kill_exactly_equivalent_survivors() -> None:
    evidence = {
        DENSE: _dense(),
        PF0: _candidate(PF0),
        PF1: _candidate(PF1),
        PF2: _candidate(PF2),
    }
    out = prune_action_level(evidence, root_level=160)
    assert out["survivors"] == sorted((PF0, PF1, PF2))
    assert out["selected_candidate"] is None
    assert out["next_level"] == 320
    assert out["fallback_used"] is False


def test_320_equivalent_survivors_escalate_to_640_without_fallback() -> None:
    evidence = {DENSE: _dense(), PF0: _candidate(PF0), PF1: _candidate(PF1)}
    out = prune_action_level(evidence, root_level=320, prior_eligible_ids=(PF0, PF1))
    assert set(out["survivors"]) == {PF0, PF1}
    assert out["selected_candidate"] is None
    assert out["next_level"] == 640
    assert out["fallback_used"] is False


def test_640_residual_equivalence_uses_pf0_conservative_fallback() -> None:
    evidence = {DENSE: _dense(), PF0: _candidate(PF0), PF1: _candidate(PF1)}
    out = prune_action_level(evidence, root_level=640, prior_eligible_ids=(PF0, PF1))
    assert out["selected_candidate"] == PF0
    assert out["fallback_used"] is True
    assert out["next_level"] is None


def test_cost_cascade_uses_10_percent_bands_then_exact_branch_minimum() -> None:
    evidence = {
        DENSE: _dense(),
        PF0: _candidate(PF0, nodes=100.0, seconds=2.0, branches=3.0),
        PF1: _candidate(PF1, nodes=109.0, seconds=2.19, branches=2.5),
        PF2: _candidate(PF2, nodes=111.0, seconds=1.0, branches=1.0),
    }
    out = prune_action_level(evidence, root_level=160)
    # PF2 is removed at rank4 (>110 nodes); PF0/PF1 survive rank4+5, then PF1
    # wins exact branch-count rank6. It is still only a 160 survivor, not final.
    assert out["survivors"] == [PF1]
    assert out["selected_candidate"] is None
    assert out["next_level"] == 320
    ranks = [row["rank"] for row in out["trace"]]
    assert ranks == [1, 2, 3, 4, 5, 6]


def test_pruned_pf0_rerun_as_control_does_not_regain_winner_eligibility() -> None:
    evidence = {
        DENSE: _dense(),
        PF0: _candidate(PF0),
        PF1: _candidate(PF1),
        PF2: _candidate(PF2),
    }
    out = prune_action_level(evidence, root_level=320, prior_eligible_ids=(PF1, PF2))
    assert PF0 not in out["survivors"]
    assert PF0 in out["mandatory_next_level_execution"]
    assert PF0 in out["control_only_noneligible"]


def test_dense_referee_learning_failure_blocks_level() -> None:
    evidence = {DENSE: _dense(False), PF0: _candidate(PF0)}
    out = prune_action_level(evidence, root_level=160)
    assert out["status"] == "BLOCKED"
    assert out["reason"] == "DENSE_REFEREE_LEARNING_GATE_FAILURE"


def test_candidate_learning_failure_is_removed_at_rank1() -> None:
    evidence = {
        DENSE: _dense(),
        PF0: _candidate(PF0, learning=False),
        PF1: _candidate(PF1),
    }
    out = prune_action_level(evidence, root_level=160)
    assert out["survivors"] == [PF1]
    assert out["trace"][0]["survivors"] == [PF1]


def test_pairing_length_mismatch_fails_closed() -> None:
    bad = CandidateSelectionEvidence(
        candidate_id=PF1,
        learning_gate_pass=True,
        domains={
            "TRUE_HEADS_UP": _domain(0.0, 0.0, n=63),
            "THREE_HANDED": _domain(0.0, 0.0, n=64),
        },
    )
    evidence = {DENSE: _dense(), PF0: _candidate(PF0), PF1: bad}
    with pytest.raises(ValueError, match="paired sample"):
        prune_action_level(evidence, root_level=160)
