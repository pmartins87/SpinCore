from __future__ import annotations

from pathlib import Path
import subprocess


LONG = Path("tools/run_3h_semantic_long_10115_10315.py")
POST = Path("tools/build_3h_semantic_postlong_policy_10315.py")
SPEC = Path("tools/build_3h_semantic_stratified_specialist_10315.py")
MULTI = Path("tools/audit_3h_semantic_stratified_specialist_multiseed_10315.py")
POST_RUN = Path("tools/run_3h_semantic_postlong_10315.sh")
DC1_EVAL = Path("tools/evaluate_deepcrusher_dc1_semantic_candidate_10315.py")
DC1_RUN = Path("tools/run_deepcrusher_dc1_semantic_postlong_10315_5k.sh")

LONG_SCHEMA = "SPINCORE_3H_SEMANTIC_LONG_ENSEMBLE_V1"
SPEC_SCHEMA = "SPINCORE_3H_SEMANTIC_STRATIFIED_STRONG_SPECIALIST_MOE_V1"


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_postlong_consumers_bind_exact_long_ensemble_schema() -> None:
    long = text(LONG)
    assert f'ENSEMBLE_SCHEMA="{LONG_SCHEMA}"' in long
    for path in (POST, SPEC, MULTI):
        body = text(path)
        assert LONG_SCHEMA in body
        assert "SPINCORE_3H_SEMANTIC_RESEARCH_ENSEMBLE_V1" not in body
    runner = text(POST_RUN)
    assert f'assert p["schema"]=="{LONG_SCHEMA}"' in runner


def test_postlong_specialist_contract_reaches_10315_dc1_evaluator() -> None:
    spec = text(SPEC)
    multi = text(MULTI)
    evaluator = text(DC1_EVAL)
    assert f'OUT_SCHEMA = "{SPEC_SCHEMA}"' in spec
    assert f'SPECIALIST_SCHEMA="{SPEC_SCHEMA}"' in multi
    assert f'SEMANTIC_STRATIFIED_SPECIALIST_MOE_SCHEMA="{SPEC_SCHEMA}"' in evaluator
    assert "SEMANTIC_COMPLETED_ITERATION=10315" in evaluator
    assert '"semantic_completed_iteration": FINAL_ITERATION' in spec
    assert '"semantic_completed_iteration":FINAL_ITERATION' in multi


def test_new_postlong_shell_runners_parse() -> None:
    for path in (POST_RUN, DC1_RUN):
        subprocess.run(["bash", "-n", str(path)], check=True)
