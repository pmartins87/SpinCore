from __future__ import annotations

"""Minimal Windows compatibility shim for Python's Unix-only ``resource`` module.

This file exists only so the frozen R7.5.3C x16 Windows worker can import the
shared Phase-2 stage module unchanged. The stage uses ``resource`` exclusively
for peak-RSS telemetry; none of these values affect RNG, sampling, model state,
training budgets, gates, or decisions.
"""

import ctypes
from ctypes import wintypes
from collections import namedtuple

RUSAGE_SELF = 0
_RUsage = namedtuple("_RUsage", ["ru_maxrss"])


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_psapi = ctypes.WinDLL("psapi", use_last_error=True)

# Explicit 64-bit-safe signatures are required. Without these declarations,
# ctypes defaults GetCurrentProcess() to c_int, which truncates/sign-extends the
# pseudo HANDLE on 64-bit Windows and causes GetProcessMemoryInfo to fail.
_kernel32.GetCurrentProcess.argtypes = []
_kernel32.GetCurrentProcess.restype = wintypes.HANDLE
_psapi.GetProcessMemoryInfo.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
    wintypes.DWORD,
]
_psapi.GetProcessMemoryInfo.restype = wintypes.BOOL


def getrusage(who: int):
    if int(who) != RUSAGE_SELF:
        raise ValueError("Windows compatibility shim supports only RUSAGE_SELF")
    counters = _PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    process = _kernel32.GetCurrentProcess()
    ok = _psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
    if not ok:
        err = ctypes.get_last_error()
        raise OSError(err, "GetProcessMemoryInfo failed")
    # Shared stage code multiplies ru_maxrss by 1024 on non-macOS platforms,
    # matching Linux's KiB convention. Return KiB here to preserve that contract.
    return _RUsage(ru_maxrss=int(counters.PeakWorkingSetSize // 1024))
