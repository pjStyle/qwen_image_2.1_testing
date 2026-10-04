"""Lightweight process memory sampling for generation telemetry."""

from __future__ import annotations

import ctypes
import os
import threading
from pathlib import Path


def current_process_rss_bytes() -> int | None:
    """Return current resident memory on Windows and Linux when available."""
    if os.name == "nt":
        try:
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
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

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            get_memory_info = psapi.GetProcessMemoryInfo
            get_memory_info.argtypes = (
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            )
            get_memory_info.restype = wintypes.BOOL
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            if get_memory_info(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return counters.WorkingSetSize
        except (AttributeError, OSError, TypeError, ValueError):
            return None
    elif os.name == "posix":
        try:
            resident_pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
        except (IndexError, OSError, ValueError):
            return None
    return None


class PeakRssSampler:
    """Sample current process RSS during a task and retain its observed peak."""

    def __init__(self, interval_seconds: float = 0.1):
        self.interval_seconds = interval_seconds
        self.peak_bytes: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        rss = current_process_rss_bytes()
        if rss is not None:
            self.peak_bytes = max(self.peak_bytes or 0, rss)

    def _monitor(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def __enter__(self) -> PeakRssSampler:
        self._sample()
        self._thread = threading.Thread(target=self._monitor, name="memory-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()
