from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "r7_5_4a_postflop_160_evaluate.yml"
TRAINING_SHA = "457996944f76e9f1fa0475691df978f450259641"
EVALUATOR_SHA = "4752a951e53c6f195fb12676a417a0b690c8e4cf"


def test_r7_5_4a_160_evaluator_is_completion_triggered_and_both_shas_are_immutable() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'workflows: ["SpinCore R7.5.4A postflop 160"]' in text
    assert "types: [completed]" in text
    assert "github.event.workflow_run.conclusion == 'success'" in text
    assert f"TRAINING_SHA: '{TRAINING_SHA}'" in text
    assert f"EVALUATOR_SHA: '{EVALUATOR_SHA}'" in text
    assert "github.event.workflow_run.head_sha" in text
    assert "ref: ${{ env.EVALUATOR_SHA }}" in text
    assert 'test "$evaluator_sha" = "$EVALUATOR_SHA"' in text
    assert "ref: ${{ needs.gate.outputs.evaluator_sha }}" in text


def test_r7_5_4a_160_evaluator_has_exact_frozen_matrix_and_prune_only_enforcement() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    for candidate in (
        "PF0_CONTROL_33_75_AI",
        "PF1_33_50_75_AI",
        "PF2_33_50_75_100_AI",
        "PF3_COMPACT_33_66_100_AI",
        "PF4_CRUSHER_COMPACT_40_66_100_AI",
        "PF_DENSE_REFERENCE",
    ):
        assert candidate in text
    assert "training_seed: [1737995611, 645939859, 1311335590]" in text
    assert "evaluation_seed: [1817694185, 1617273629]" in text
    assert "tools/r7_5_4a_160_dense_cache_worker.py" in text
    assert "tools/r7_5_4a_160_candidate_cell_worker.py" in text
    assert "tools/r7_5_4a_160_cross_seed_worker.py" in text
    assert "tools/r7_5_4a_160_aggregate.py" in text
    assert "validation/R7_5_4A_160_RESULT.json" in text
    assert "selection['selected_candidate'] is None" in text
    assert "selection['next_level']==320" in text
    assert "result['ready_for_tables'] is False" in text
    assert "result['production_training_authorized'] is False" in text


def test_r7_5_4a_160_result_is_persisted_before_enforcement() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    aggregate = text.index("  aggregate:")
    persist = text.index("  persist:")
    enforce = text.index("  enforce:")
    assert aggregate < persist < enforce
    assert "Upload durable 160 aggregate result before enforcement" in text
    assert "git push origin HEAD:main" in text
