#!/usr/bin/env python3
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SRC=(ROOT/"tools"/"run_3h_semantic_long_block.py").read_text(encoding="utf-8")
PAR=(ROOT/"tools"/"run_3h_semantic_long_block_parallel.py").read_text(encoding="utf-8")


def test_generic_3h_long_block_preserves_frozen_training_contract():
    assert 'ROOTS_PER_ITERATION=pilot.ROOTS_PER_ITERATION' in SRC
    assert 'MEMBER_STEPS=pilot.MEMBER_STEPS' in SRC
    assert 'ENSEMBLE_SIZE=8' in SRC
    assert 'MILESTONE_INTERVAL=500' in SRC
    assert 'pilot.collect_roots(' in SRC
    assert 'pilot.fit_ensemble(' in SRC
    assert 'reference_long.safety_guard' in SRC
    assert 'hu_training_performed":False' in SRC
    assert 'average_policy_training_performed":False' in SRC


def test_generic_3h_long_block_binds_source_endpoint_and_is_resumable():
    for token in (
        'source_endpoint_resume_sha256',
        'source_endpoint_ensemble_sha256',
        'source_endpoint_report_sha256',
        'source 3H resume hash does not match PASS report',
        'source 3H ensemble hash does not match PASS report',
        'frozen 3H source endpoint changed during continuation',
        'save_resume(',
        'restore_local_resume(',
    ):
        assert token in SRC


def test_generic_3h_root_accounting_survives_resume():
    assert 'block_roots=(completed-start_iteration)*ROOTS_PER_ITERATION' in SRC
    assert 'block_roots+=int(tree["roots"])' in SRC
    assert 'if block_roots!=expected_roots:' in SRC


def test_generic_parallel_wrapper_only_replaces_fit_path():
    assert 'import run_3h_semantic_long_block as serial_long' in PAR
    assert 'pilot.fit_ensemble=_parallel_fit_ensemble' in PAR
    assert 'PackedSemanticAdvantageReservoir.build' in PAR
    assert 'ParallelSemanticEnsembleFitter' in PAR
