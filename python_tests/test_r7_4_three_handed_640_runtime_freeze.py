from pathlib import Path


def test_r7_4_three_handed_640_runtime_is_explicitly_frozen():
    workflow = Path(".github/workflows/r7_4_three_handed_640_confirmation.yml").read_text(
        encoding="utf-8"
    )
    assert "python-version: '3.11'\n" not in workflow
    assert workflow.count("python-version: '3.11.15'") == 7
    assert workflow.count("'torch==2.13.0+cpu'") == 7
    assert workflow.count("torch.__version__=='2.13.0+cpu'") == 7
    assert "https://download.pytorch.org/whl/cpu torch\n" not in workflow
