"""Progress bar implementations for parallel execution.

This module provides:
- Simple single progress bar for basic pmap usage
- Multi-job progress bars showing per-CPU progress
- Joblib callback integration for progress updates
"""
from __future__ import annotations

import contextlib
import multiprocessing
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import joblib
from joblib import Parallel, delayed
from rich.live import Live
from rich.progress import (
    Progress,
    BarColumn,
    MofNCompleteColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

from .loguru_routing import (
    LoguruConfig,
    find_loguru_names,
    strip_loguru_from_globals,
    reinject_loguru,
    make_worker_wrapper,
    redirect_loguru_to_live,
    log_consumer,
)
from .progress_styles import (
    create_progress_columns,
    create_progress_table,
    make_job_description,
)

# Progress bar refresh rate (Hz)
DEFAULT_REFRESH_RATE = 4 * 4

# Thread polling interval for progress updates (seconds)
PROGRESS_POLL_INTERVAL = 0.1


@dataclass
class ParallelMode:
    """Configuration for parallel execution mode."""
    using_threads: bool
    log_queue: Any  # multiprocessing.Queue | None
    wrapped_func: Callable
    stripped_names: set[str]


def prepare_parallel_mode(f: Callable, prefer: str | None) -> ParallelMode:
    """Prepare function and logging for the selected parallel backend."""
    using_threads = prefer == 'threads'

    if using_threads:
        return ParallelMode(
            using_threads=True,
            log_queue=None,
            wrapped_func=f,
            stripped_names=set()
        )

    stripped_names = find_loguru_names(f)
    strip_loguru_from_globals(f, stripped_names)
    log_queue = multiprocessing.Manager().Queue()
    config = LoguruConfig(stripped_names, log_queue)

    return ParallelMode(
        using_threads=False,
        log_queue=log_queue,
        wrapped_func=make_worker_wrapper(f, config),
        stripped_names=stripped_names
    )


@contextlib.contextmanager
def progress_with_live(desc: str, total: int, disable: bool = False):
    """Progress bar with Live display that allows printing above it."""
    if disable:
        yield None, None
        return

    progress = Progress(
        TextColumn("[progress.percentage]{task.description} {task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )
    task_id = progress.add_task(desc, total=total)

    class RichBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, live=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._live = live

        def __call__(self, out):
            progress.update(task_id, advance=self.batch_size)
            if self._live:
                self._live.refresh()
            return super().__call__(out)

    old_callback = joblib.parallel.BatchCompletionCallBack

    def make_callback_class(live):
        class Callback(RichBatchCompletionCallback):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, live=live, **kwargs)
        return Callback

    with Live(progress, refresh_per_second=16, transient=False) as live:
        joblib.parallel.BatchCompletionCallBack = make_callback_class(live)
        try:
            yield progress, live
        finally:
            joblib.parallel.BatchCompletionCallBack = old_callback
            progress.update(task_id, completed=total)


def sequential_map(f: Callable, arr: list, desc: str, disable: bool) -> list:
    """Sequential map with progress bar for n_jobs=1 case."""
    if disable:
        return [f(item) for item in arr]

    results = []
    progress = Progress(
        TextColumn("[progress.percentage]{task.description} {task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )
    task_id = progress.add_task(desc, total=len(arr))

    with Live(progress, refresh_per_second=16, transient=False) as live:
        with redirect_loguru_to_live(live):
            for item in arr:
                results.append(f(item))
                progress.update(task_id, advance=1)
    return results


def run_with_simple_bar(mode: ParallelMode, arr: list, n_jobs: int, batch_size, disable_tqdm: bool, desc: str, kwargs: dict) -> list:
    """Run parallel execution with a simple single progress bar."""
    with progress_with_live(desc, total=len(arr), disable=disable_tqdm) as (progress, live):
        with log_consumer(mode.log_queue, live):
            if mode.using_threads:
                with redirect_loguru_to_live(live):
                    results = Parallel(n_jobs=n_jobs, batch_size=batch_size, **kwargs)(
                        delayed(mode.wrapped_func)(i) for i in arr
                    )
            else:
                results = Parallel(n_jobs=n_jobs, batch_size=batch_size, **kwargs)(
                    delayed(mode.wrapped_func)(i) for i in arr
                )
    return results


def run_with_job_bars(mode: ParallelMode, arr: list, n_jobs: int, batch_size, disable_tqdm: bool, desc: str, total_tasks: int, total_cpus: int, kwargs: dict) -> list:
    """Run parallel execution with per-job progress bars and CPU info."""
    job_progress, overall_progress = create_progress_columns(disable_tqdm)

    task_id = job_progress.add_task(desc, total=total_tasks)
    overall_task_id = overall_progress.add_task(
        Text.assemble(("", "dim blue"), (" Total", "bold white")),
        total=total_tasks
    )

    class DynamicProgressTable:
        """Renderable that regenerates the progress table on each render."""
        def __rich__(self):
            completed = int(job_progress.tasks[task_id].completed)
            return create_progress_table(
                job_progress,
                overall_progress,
                total_cpus,
                completed,
                total_tasks
            )

    with contextlib.ExitStack() as stack:
        live = stack.enter_context(Live(DynamicProgressTable(), refresh_per_second=DEFAULT_REFRESH_RATE, transient=False))
        if mode.log_queue:
            stack.enter_context(log_consumer(mode.log_queue, live))
        stack.enter_context(rich_joblib_adaptive(job_progress, overall_progress, task_id, overall_task_id, total_cpus))
        if mode.using_threads:
            stack.enter_context(redirect_loguru_to_live(live))
        results = Parallel(n_jobs=n_jobs, batch_size=batch_size, **kwargs)(
            delayed(mode.wrapped_func)(i) for i in arr
        )

    return results


@contextlib.contextmanager
def rich_joblib_adaptive(job_progress, overall_progress, overall_job_task_id, overall_progress_task_id, total_cpus):
    """Enhanced context manager for joblib with styled progress bars."""
    class RichBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        _job_counter = 0
        _job_counter_lock = threading.Lock()
        _completed_tasks = 0

        @classmethod
        def reset_counters(cls):
            cls._job_counter = 0
            cls._completed_tasks = 0

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.batch_start_time = time.time()
            self.first_batch = True
            self.completed_jobs = 0
            self.active_jobs = {}
            self.job_times = []
            self.avg_job_time = None

            self._stop_event = threading.Event()
            self._thread = threading.Thread(target=self._update_progress)
            self._thread.daemon = True
            self._thread.start()

        @classmethod
        def get_next_job_number(cls):
            with cls._job_counter_lock:
                cls._job_counter += 1
                return cls._job_counter

        @property
        def active_jobs_count(self):
            return len(self.active_jobs)

        def _update_progress(self):
            while not self._stop_event.is_set():
                if self.avg_job_time:
                    current_time = time.time()
                    for job_idx, (job_task_id, start_time, _) in list(self.active_jobs.items()):
                        elapsed = current_time - start_time
                        progress = min(99, int(100 * elapsed / self.avg_job_time))
                        if progress >= 99:
                            job_progress.remove_task(job_task_id)
                            self.active_jobs.pop(job_idx)
                            self.__class__._completed_tasks += 1
                        else:
                            job_progress.update(job_task_id, completed=progress)
                    job_progress.refresh()
                time.sleep(PROGRESS_POLL_INTERVAL)

        def __call__(self, *args, **kwargs):
            current_time = time.time()

            if self.batch_size > 0:
                elapsed = current_time - self.batch_start_time
                batch_avg = elapsed / self.batch_size
                self.job_times.append(batch_avg)
                self.avg_job_time = sum(self.job_times) / len(self.job_times)
                self.batch_start_time = current_time

            if self.first_batch:
                job_progress.update(overall_job_task_id, total=job_progress.tasks[overall_job_task_id].total)
                overall_progress.update(overall_progress_task_id, total=job_progress.tasks[overall_progress_task_id].total)
                self.first_batch = False

            for job_idx in list(self.active_jobs.keys()):
                if job_idx < self.completed_jobs:
                    job_task_id, _, _ = self.active_jobs.pop(job_idx)
                    job_progress.remove_task(job_task_id)

            new_job_idx = self.completed_jobs
            job_number = self.get_next_job_number()

            new_job_task_id = job_progress.add_task(
                make_job_description(job_number, total_cpus, self.active_jobs_count + 1, estimating=self.first_batch),
                total=100,
                completed=0
            )
            self.active_jobs[new_job_idx] = (new_job_task_id, current_time, job_number)

            job_progress.update(overall_job_task_id, advance=self.batch_size)
            overall_progress.update(overall_progress_task_id, advance=self.batch_size)

            self.completed_jobs += self.batch_size
            return super().__call__(*args, **kwargs)

        def stop(self):
            self._stop_event.set()
            self._thread.join(timeout=1.0)

    RichBatchCompletionCallback.reset_counters()

    callback = RichBatchCompletionCallback
    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    current_callback_instance = None

    class WrappedCallback(callback):
        def __init__(self, *args, **kwargs):
            nonlocal current_callback_instance
            super().__init__(*args, **kwargs)
            current_callback_instance = self

    joblib.parallel.BatchCompletionCallBack = WrappedCallback
    try:
        yield
    finally:
        if current_callback_instance is not None:
            current_callback_instance.stop()
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        job_progress.update(overall_job_task_id, completed=job_progress.tasks[overall_job_task_id].total)
        overall_progress.update(overall_progress_task_id, completed=overall_progress.tasks[overall_progress_task_id].total)
