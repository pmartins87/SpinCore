#!/usr/bin/env python3
"""Thin gate wrapper around validate_lt2_concurrent_iteration.py.

The candidate report contains one additional timing-only field,
`concurrent_fit_wall_seconds`.  It must not participate in semantic parity.
Keep the main validator frozen and patch only the timing normalizer here.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "tools" / "validate_lt2_concurrent_iteration.py"
spec = importlib.util.spec_from_file_location("spincore_lt2_concurrent_iteration_validator", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load base validator")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def strip_timing(value):
    timing_keys = {
        "seconds",
        "tree_seconds",
        "seconds_per_root",
        "advantage_fit_seconds",
        "advantage_fit_profile",
        "concurrent_fit_wall_seconds",
    }
    if isinstance(value, dict):
        return {
            key: strip_timing(item)
            for key, item in value.items()
            if key not in timing_keys
        }
    if isinstance(value, list):
        return [strip_timing(item) for item in value]
    return value


module.strip_timing = strip_timing

if __name__ == "__main__":
    raise SystemExit(module.main())
