#!/usr/bin/env -S uv run
"""Test pmap with all combinations of output methods and execution modes."""
import time
from loguru import logger
import rich
from mlh.parallel import pmap


def worker_all_prints(i: int) -> int:
    """Worker that uses all output methods."""
    print(f"print: item {i}")
    rich.print(f"[bold cyan]rich: item {i}[/bold cyan]")
    logger.info(f"loguru: item {i}")
    time.sleep(4)
    return i * 2


def run_test(name: str, **kwargs):
    """Run a test with given kwargs and print results."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"kwargs: {kwargs}")
    print(f"{'='*60}")

    results = pmap(worker_all_prints, range(3), **kwargs)

    print(f"\nResults: {results}")
    assert results == [0, 2, 4], f"Expected [0, 2, 4], got {results}"
    print("PASSED")


if __name__ == "__main__":
    # Test all combinations
    tests = [
        # Processes (default backend)
        ("Processes + Simple Bar", {"n_jobs": 2}),
        ("Processes + Job Bars", {"n_jobs": 2, "show_job_bars": True}),

        # Threads
        ("Threads + Simple Bar", {"n_jobs": 2, "prefer": "threads"}),
        ("Threads + Job Bars", {"n_jobs": 2, "prefer": "threads", "show_job_bars": True}),

        # Sequential (n_jobs=1)
        ("Sequential (n_jobs=1)", {"n_jobs": 1}),

        # With progress disabled
        ("Processes + Disabled Progress", {"n_jobs": 2, "disable_tqdm": True}),
        ("Job Bars + Disabled Progress", {"n_jobs": 2, "show_job_bars": True, "disable_tqdm": True}),
    ]

    for name, kwargs in tests:
        run_test(name, **kwargs)

    print(f"\n{'='*60}")
    print("ALL TESTS PASSED!")
    print(f"{'='*60}")
