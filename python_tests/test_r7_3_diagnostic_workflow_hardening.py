from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/r7_3_diagnostic.yml"


def test_r7_3_diagnostic_is_read_only_and_artifact_backed():
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "permissions:\n  contents: read" in text
    assert "numpy==2.3.5 pytest" in text
    assert "uses: actions/upload-artifact@v4" in text
    assert "path: validation/R7_3_DIAGNOSTIC_640.json" in text
    assert "if-no-files-found: error" in text
    assert "retention-days: 90" in text

    for forbidden in (
        "contents: write",
        "git push",
        "git pull",
        "git commit",
        "git add",
        "git config",
    ):
        assert forbidden not in text


def test_r7_3_frozen_gate_and_diagnostic_commands_remain_present():
    text = WORKFLOW.read_text(encoding="utf-8")

    required = (
        "--iterations 2",
        "--roots-per-iteration 8",
        "--advantage-steps 16",
        "--advantage-max-steps-per-iteration 32",
        "--advantage-fit-target 10",
        "--policy-chunk-steps 16",
        "--policy-max-steps 32",
        "--policy-fit-target 10",
        "--batch-size 32",
        "--audit-size 64",
        "--cross-seed-per-seed 64",
        "--reservoir-capacity 10000",
        "--out validation/R7_3_DIAGNOSTIC_640.json",
        "--strict",
        "code='${{ steps.r73.outputs.exit_code }}'",
        'exit "$code"',
    )
    for token in required:
        assert token in text
