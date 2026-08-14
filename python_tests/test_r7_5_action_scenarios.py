from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import r7_4_stability_pilot_worker as r74
from spincore.r7_5_action_scenarios import action_scenario_cycle, scenario_descriptor


def test_r7_5_action_scenarios_are_exact_r7_4_mechanical_copy() -> None:
    for domain in ("TRUE_HEADS_UP", "THREE_HANDED"):
        old = [r74._scenario_descriptor(ep) for ep in r74._scenario_cycle(domain)]
        new = [scenario_descriptor(ep) for ep in action_scenario_cycle(domain)]
        assert new == old
    assert len(action_scenario_cycle("TRUE_HEADS_UP")) == 6
    assert len(action_scenario_cycle("THREE_HANDED")) == 15
