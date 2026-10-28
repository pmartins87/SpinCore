#!/usr/bin/env python3
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SRC=(ROOT/"tools"/"run_hu_semantic_ens8_long_block.py").read_text(encoding="utf-8")


def test_generic_hu_long_block_preserves_semantic_teacher_contract():
    assert 'ROOTS_PER_UPDATE = base.ROOTS_PER_UPDATE' in SRC
    assert 'MEMBER_STEPS = base.MEMBER_STEPS' in SRC
    assert 'ENSEMBLE_SIZE = base.ENSEMBLE_SIZE' in SRC
    assert 'SemanticParallelRootExecutor' in SRC
    assert 'base.fit_semantic_ensemble(fitter)' in SRC
    assert 'base.make_jobs(' in SRC
    assert 'expand_v1_state_with_zero_semantics' not in SRC
    assert 'migration_parity(' not in SRC


def test_generic_hu_long_block_is_cross_domain_fail_closed():
    assert '"three_handed_training_performed": False' in SRC
    assert '"average_policy_training_performed": False' in SRC
    assert '"strategy_reservoir_collection_performed": False' in SRC
    assert 'force_domain' not in SRC or 'base.make_jobs' in SRC
    assert '_collect_policy_from_episodes' not in SRC


def test_generic_hu_long_block_binds_source_endpoint_hashes():
    for token in (
        'source_endpoint_resume_sha256',
        'source_endpoint_ensemble_sha256',
        'source_endpoint_report_sha256',
        'source HU resume hash does not match PASS report',
        'source HU ensemble hash does not match PASS report',
        'frozen HU source endpoint changed during long continuation',
    ):
        assert token in SRC


def test_cumulative_interval_is_not_hardcoded_into_generic_runner():
    assert '--start-cumulative-updates' in SRC
    assert '--target-cumulative-updates' in SRC
    assert 'for cumulative_update in range(completed_updates + 1, target_updates + 1)' in SRC
