#!/usr/bin/env python3
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PLAN=json.loads((ROOT/"contracts"/"three_month_training_plan_v1.json").read_text(encoding="utf-8"))
ORCH=(ROOT/"tools"/"run_three_month_training_v1.py").read_text(encoding="utf-8")


def test_three_month_plan_has_exact_alternating_endpoint_chain():
    stages=PLAN["stages"]
    assert len(stages)==21
    assert stages[0]=={
        "id":"3h_block_01","cycle":1,"domain":"3H",
        "start":17315,"target":19315,"roots":128000,"mode":"specific"
    }
    assert stages[-1]["id"]=="3h_block_11"
    assert stages[-1]["start"]==37315
    assert stages[-1]["target"]==39315

    last_3h=17315
    last_hu=469
    expected_domain="3H"
    hu_blocks=1  # current prerequisite already running
    three_blocks=0
    for stage in stages:
        assert stage["domain"]==expected_domain
        if stage["domain"]=="3H":
            assert stage["start"]==last_3h
            assert stage["target"]-stage["start"]==2000
            assert stage["roots"]==128000
            last_3h=stage["target"]
            three_blocks+=1
            expected_domain="HU" if three_blocks<=10 else "END"
        else:
            assert stage["start"]==last_hu
            assert stage["target"]-stage["start"]==469
            assert stage["roots"]==128037
            last_hu=stage["target"]
            hu_blocks+=1
            expected_domain="3H"
    assert expected_domain=="END"
    assert hu_blocks==11
    assert three_blocks==11
    assert last_hu==5159
    assert last_3h==39315


def test_three_month_compute_budget_totals_are_exact():
    t=PLAN["target_compute_budget"]
    assert t["hu_cumulative_updates_at_end"]==469*11
    assert t["hu_new_roots_total_since_10105"]==469*273*11
    assert t["hu_virtual_seed_iteration_end"]==10105+469*11
    assert t["three_handed_additional_iterations"]==2000*11
    assert t["three_handed_new_roots"]==2000*64*11
    assert t["three_handed_iteration_end"]==17315+2000*11


def test_plan_is_compute_budget_not_results_driven_selection():
    s=PLAN["scheduling"]
    assert s["policy"]=="ALTERNATE_ONE_FIXED_ROOT_BLOCK_PER_DOMAIN"
    assert s["concurrency"]=="ONE_HEAVY_TRAINER_AT_A_TIME"
    assert s["calendar_rule"]=="COMPUTE_BUDGET_NOT_WALL_CLOCK_STOP"
    assert s["strength_selection_during_plan"] is False


def test_orchestrator_is_fail_closed_and_does_not_self_update():
    assert "git pull" not in ORCH
    assert "ensure_no_external_trainer()" in ORCH
    assert "ensure_resources()" in ORCH
    assert "check_support_blobs(plan)" in ORCH
    assert "ensure_clean_repo()" in ORCH
    assert "STOPPED_STAGE_FAILURE" in ORCH
    assert "SPINCORE_THREE_MONTH_TRAINING_PLAN_PASS" in ORCH
    assert "--dry-run" in ORCH
