from __future__ import annotations

"""Minimal Windows compatibility shim for Python's Unix-only ``resource`` module.

This file exists only so the frozen R7.5.3C x16 Windows worker can import the
shared Phase-2 stage module unchanged.  The stage uses ``resource`` exclusively
for peak-RSS telemetry; none of these values affect RNG, sampling, model state,
training budgets, gates, or decisions.
"""

import ctypes
from collections import namedtuple

RUSAGE_SELF = 0
_RUsage = namedtuple("_RUsage", ["ru_maxrss"])


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def getrusage(who: int):
    if int(who) != RUSAGE_SELF:
        raise ValueError("Windows compatibility shim supports only RUSAGE_SELF")
    counters = _PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    process = ctypes.windll.kernel32.GetCurrentProcess()
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(
        process,
        ctypes.byref(counters),
        counters.cb,
    )
    if not ok:
        raise OSError("GetProcessMemoryInfo failed")
    # Shared stage code multiplies ru_maxrss by 1024 on non-macOS platforms,
    # matching Linux's KiB convention. Return KiB here to preserve that contract.
    return _RUsage(ru_maxrss=int(counters.PeakWorkingSetSize // 1024))
