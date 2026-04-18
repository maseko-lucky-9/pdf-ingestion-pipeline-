#!/usr/bin/env python3
"""Memory profiling utilities for benchmarks."""

import tracemalloc
import psutil
import time
from contextlib import contextmanager
from typing import Dict, Optional


class MemoryProfiler:
    """Process-specific memory tracking."""

    def __init__(self):
        self.tracemalloc_started = False
        self.start_time = None
        self.start_rss = None
        self.peak_tracemalloc = 0
        self.peak_rss = 0

    def start(self):
        """Start memory profiling."""
        tracemalloc.start()
        self.tracemalloc_started = True
        self.start_time = time.perf_counter()
        self.start_rss = psutil.Process().memory_info().rss
        self.peak_tracemalloc = 0
        self.peak_rss = self.start_rss

    def checkpoint(self) -> Dict[str, float]:
        """Get current memory stats."""
        if not self.tracemalloc_started:
            return {}

        current, peak = tracemalloc.get_traced_memory()
        current_rss = psutil.Process().memory_info().rss

        self.peak_tracemalloc = max(self.peak_tracemalloc, peak)
        self.peak_rss = max(self.peak_rss, current_rss)

        return {
            "tracemalloc_current_mb": current / 1024**2,
            "tracemalloc_peak_mb": peak / 1024**2,
            "rss_current_mb": current_rss / 1024**2,
            "rss_delta_mb": (current_rss - self.start_rss) / 1024**2,
            "elapsed_s": time.perf_counter() - self.start_time if self.start_time else 0
        }

    def stop(self) -> Dict[str, float]:
        """Stop profiling and return final stats."""
        if not self.tracemalloc_started:
            return {}

        stats = self.checkpoint()
        tracemalloc.stop()
        self.tracemalloc_started = False

        stats["peak_tracemalloc_mb"] = self.peak_tracemalloc / 1024**2
        stats["peak_rss_mb"] = self.peak_rss / 1024**2

        return stats


@contextmanager
def profile_memory(name: str = "operation"):
    """Context manager for memory profiling."""
    profiler = MemoryProfiler()
    profiler.start()

    try:
        yield profiler
    finally:
        stats = profiler.stop()
        print(f"[{name}] Peak memory: {stats.get('peak_tracemalloc_mb', 0):.1f} MB (tracemalloc), "
              f"{stats.get('peak_rss_mb', 0):.1f} MB (RSS)")


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage without profiling."""
    process = psutil.Process()
    mem_info = process.memory_info()

    return {
        "rss_mb": mem_info.rss / 1024**2,
        "vms_mb": mem_info.vms / 1024**2,
        "percent": process.memory_percent()
    }