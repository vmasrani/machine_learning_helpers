from __future__ import annotations

import contextlib
import io
import multiprocessing
import sys
import threading
import time
import warnings
from queue import Queue
from typing import Any, Callable, Iterable

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from sklearn.model_selection import GroupKFold

from .progress_styles import (
    create_progress_columns,
    create_progress_table,
    make_job_description,
)

# Suppress specific FutureWarning about 'DataFrame.swapaxes'
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="'DataFrame.swapaxes' is deprecated.*"
)

# Progress bar refresh rate (Hz) - tuned to prevent flickering
DEFAULT_REFRESH_RATE = 4*4

# Thread polling interval for progress updates (seconds)
PROGRESS_POLL_INTERVAL = 0.1


@contextlib.contextmanager
def rich_joblib(progress, task_id, live=None):
    """Context manager to patch joblib to report into rich progress bar and capture outputs"""
    class RichBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, out):
            # Extract outputs and print them above the live display
            if live is not None and out is not None:
                with contextlib.suppress(Exception):
                    batch_results = out.result()
                    for result, output in batch_results:
                        if output:
                            # Use live.console.print() to print above the progress bar
                            live.console.print(output, style="dim cyan")
            progress.update(task_id, advance=self.batch_size)
            if live is not None:
                live.refresh()
            return super().__call__(out)


    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = RichBatchCompletionCallback  # ty:ignore[invalid-assignment]
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        progress.update(task_id, completed=progress.tasks[task_id].total)
        progress.refresh()
        progress.update(task_id, visible=False)

def safe(f: Callable) -> Callable:
    """Wrap function to catch exceptions and return error dict instead of raising."""
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            return {
                'error': str(e),
                'error_type': type(e).__name__,
                'args': args,
                'kwargs': kwargs
            }
    return wrapper


@contextlib.contextmanager
def _redirect_loguru_to_console(console):
    """Temporarily redirect loguru to use Rich console for coordinated output."""
    try:
        import loguru
        loguru.logger.remove()
        handler_id = loguru.logger.add(
            lambda m: console.print(m, end=""),
            colorize=True,
            format="{time:HH:mm:ss} | {level: <8} | {message}"
        )
        try:
            yield
        finally:
            loguru.logger.remove(handler_id)
            loguru.logger.add(sys.stderr)
    except ImportError:
        yield


def _strip_loguru_from_globals(f: Callable) -> tuple[Callable, list[str]]:
    """Remove loguru Logger instances from function globals to allow pickling.

    Returns the modified function and list of global names that were stripped.
    The wrapper will re-inject loguru in the worker process.
    """
    stripped_names = []
    try:
        from loguru._logger import Logger
        if hasattr(f, '__globals__'):
            for name, value in list(f.__globals__.items()):
                if isinstance(value, Logger):
                    stripped_names.append(name)
                    f.__globals__[name] = None  # Replace with None to allow pickling
    except ImportError:
        pass
    return f, stripped_names


def _capture_stdout_wrapper(f: Callable, loguru_global_names: list[str]) -> Callable:
    """Wrapper that captures stdout, stderr, and loguru output from function f and returns (result, output)."""
    def wrapper(*args, **kwargs):
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        # Handle loguru only if we stripped globals (process mode, not threads)
        # With threads, loguru is shared across all threads so we can't capture per-task
        loguru_configured = False
        if loguru_global_names:
            try:
                import loguru

                # Configure the module logger to write to our buffer
                loguru.logger.configure(handlers=[{
                    "sink": stderr_buffer,
                    "format": "{level}: {message}",
                    "colorize": False
                }])

                # Re-inject loguru into function globals (we stripped it before pickling)
                if hasattr(f, '__globals__'):
                    for name in loguru_global_names:
                        f.__globals__[name] = loguru.logger

                loguru_configured = True
            except ImportError:
                pass

        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                result = f(*args, **kwargs)
        finally:
            if loguru_configured:
                with contextlib.suppress(Exception):
                    import loguru
                    loguru.logger.configure(handlers=[])

        # Combine stdout and stderr, with stderr/loguru going first if present
        stderr_output = stderr_buffer.getvalue().rstrip()
        stdout_output = stdout_buffer.getvalue().rstrip()
        combined_output = '\n'.join(filter(None, [stderr_output, stdout_output]))
        return (result, combined_output)
    return wrapper

def pmap(f, arr, n_jobs=-1, disable_tqdm=False, safe_mode=False, spawn=False, batch_size='auto', **kwargs):
    arr = list(arr)  # Convert generators to list for progress tracking.
    desc = kwargs.pop('desc', 'Processing')

    if spawn:
        multiprocessing.set_start_method('spawn', force=True)

    f = safe(f) if safe_mode else f

    # Check if using threads (threads share memory, no pickling needed)
    using_threads = kwargs.get('prefer') == 'threads'

    if using_threads:
        # With threads, skip all special processing - threads share stdout/stderr
        # and the progress bar needs direct access to work properly
        f_wrapped = lambda *args, **kw: (f(*args, **kw), '')
    else:
        # With processes, strip loguru before pickling (will be re-injected in worker)
        f, loguru_global_names = _strip_loguru_from_globals(f)
        f_wrapped = _capture_stdout_wrapper(f, loguru_global_names)

    console = Console()

    progress_bar = Progress(
        TextColumn("[progress.percentage]{task.description} {task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        disable=disable_tqdm,
        transient=kwargs.pop('transient', False),
    )

    if using_threads:
        # For threads, redirect loguru through the same console as progress bar
        with _redirect_loguru_to_console(console):
            with progress_bar:
                # console.print()
                task_id = progress_bar.add_task(desc, total=len(arr))
                with rich_joblib(progress_bar, task_id, live=None):
                    results_with_output = Parallel(n_jobs=n_jobs, batch_size=batch_size, **kwargs)(delayed(f_wrapped)(i) for i in arr)
    else:
        with Live(progress_bar, console=console, refresh_per_second=DEFAULT_REFRESH_RATE) as live:
            task_id = progress_bar.add_task(desc, total=len(arr))
            with rich_joblib(progress_bar, task_id, live):
                results_with_output = Parallel(n_jobs=n_jobs, batch_size=batch_size, **kwargs)(delayed(f_wrapped)(i) for i in arr)

    # Separate results from outputs
    results = [result for result, output in results_with_output]

    return results

def pmap_df(
    f: Callable,
    df: pd.DataFrame,
    n_chunks: int = 100,
    groups: str | None = None,
    axis: int = 0,
    safe_mode: bool = False,
    **kwargs
) -> pd.DataFrame:
    # https://towardsdatascience.com/make-your-own-super-pandas-using-multiproc-1c04f41944a1
    if groups:
        n_chunks = min(n_chunks, df[groups].nunique())
        group_kfold = GroupKFold(n_splits=n_chunks)
        df_split = [df.iloc[test_index] for _, test_index in group_kfold.split(df, groups=df[groups])]
    else:
        df_split = np.array_split(df, n_chunks)
    df = pd.concat(pmap(f, df_split, safe_mode=safe_mode, **kwargs), axis=axis)  # type: ignore[assignment]
    return df


def run_async(func):
    """Run function asynchronously and return a queue for retrieving results.

    Example:
        @run_async
        def long_run(idx, val='cat'):
            for i in range(idx):
                print(i)
                time.sleep(1)
            return val

        queue = long_run(5, val='dog')
        result = queue.get()
    """
    import multiprocessing
    def func_with_queue(queue, *args, **kwargs):
        print(f'Running function {func.__name__}{args} {kwargs} ... ')
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        queue.put(result)
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')

    def wrapper(*args, **kwargs):
        queue = multiprocessing.Manager().Queue()
        process = multiprocessing.Process(target=func_with_queue, args=(queue, *args), kwargs=kwargs)
        process.start()
        return queue
    return wrapper




@contextlib.contextmanager
def rich_joblib_adaptive(job_progress, overall_progress, overall_job_task_id, overall_progress_task_id, total_cpus):
    """Enhanced context manager for joblib with styled progress bars."""
    class RichBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        _job_counter = 0
        _job_counter_lock = threading.Lock()
        _completed_tasks = 0  # Track completed tasks

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.start_time = time.time()
            self.first_batch = True
            self.completed_jobs = 0
            self.active_jobs = {}
            self.max_visible_jobs = 10
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
            """Update progress of active jobs in a separate thread."""
            while not self._stop_event.is_set():
                if self.avg_job_time:
                    current_time = time.time()
                    for job_idx, (job_task_id, start_time, _) in list(self.active_jobs.items()):
                        elapsed = current_time - start_time
                        progress = min(99, int(100 * elapsed / self.avg_job_time))
                        if progress >= 99:  # Job is nearly complete
                            job_progress.remove_task(job_task_id)
                            self.active_jobs.pop(job_idx)
                            self.__class__._completed_tasks += 1  # Increment completed tasks when job ends
                        else:
                            job_progress.update(job_task_id, completed=progress)
                    job_progress.refresh()
                time.sleep(PROGRESS_POLL_INTERVAL)

        def __call__(self, *args, **kwargs):
            current_time = time.time()

            if self.first_batch and self.batch_size > 0:
                elapsed = current_time - self.start_time
                self.avg_job_time = elapsed / self.batch_size
                job_progress.update(overall_job_task_id, total=job_progress.tasks[overall_job_task_id].total)
                overall_progress.update(overall_progress_task_id, total=job_progress.tasks[overall_progress_task_id].total)
                self.first_batch = False

            # Update active jobs
            for job_idx in list(self.active_jobs.keys()):
                if job_idx < self.completed_jobs:
                    job_task_id, _, _ = self.active_jobs.pop(job_idx)
                    job_progress.remove_task(job_task_id)

            # Add new job with styled description
            new_job_idx = self.completed_jobs
            job_number = self.get_next_job_number()

            new_job_task_id = job_progress.add_task(
                make_job_description(job_number, total_cpus, self.active_jobs_count + 1),
                total=100,
                completed=0
            )
            self.active_jobs[new_job_idx] = (new_job_task_id, current_time, job_number)

            # Update progress
            job_progress.update(overall_job_task_id, advance=self.batch_size)
            overall_progress.update(overall_progress_task_id, advance=self.batch_size)

            self.completed_jobs += self.batch_size
            return super().__call__(*args, **kwargs)

        def stop(self):
            self._stop_event.set()
            self._thread.join(timeout=1.0)

    callback = RichBatchCompletionCallback
    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    current_callback_instance = None

    class WrappedCallback(callback):
        def __init__(self, *args, **kwargs):
            nonlocal current_callback_instance
            super().__init__(*args, **kwargs)
            current_callback_instance = self

    joblib.parallel.BatchCompletionCallBack = WrappedCallback  # ty:ignore[invalid-assignment]
    try:
        yield
    finally:
        if current_callback_instance is not None:
            current_callback_instance.stop()
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        job_progress.update(overall_job_task_id, completed=job_progress.tasks[overall_job_task_id].total)
        overall_progress.update(overall_progress_task_id, completed=overall_progress.tasks[overall_progress_task_id].total)



def pmap_multi(
    f: Callable,
    arr: Iterable[Any],
    n_jobs: int = -1,
    disable_tqdm: bool = False,
    safe_mode: bool = False,
    spawn: bool = False,
    batch_size: str | int = 'auto',
    **kwargs
) -> list:
    """Parallel map with styled progress bars and CPU usage information."""
    arr = list(arr)
    desc = kwargs.pop('desc', 'Processing')
    total_tasks = len(arr)

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    total_cpus = min(n_jobs, total_tasks)

    if spawn:
        multiprocessing.set_start_method('spawn', force=True)

    f = safe(f) if safe_mode else f

    # Create styled progress bars
    job_progress, overall_progress = create_progress_columns(disable_tqdm)

    # Add tasks with styled descriptions
    task_id = job_progress.add_task(desc, total=total_tasks)
    overall_task_id = overall_progress.add_task(
        Text.assemble(("", "dim blue"), (" Total", "bold white")),
        total=total_tasks
    )

    # Track completed tasks
    completed_tasks = 0

    def update_progress_table():
        return create_progress_table(
            job_progress,
            overall_progress,
            total_cpus,
            completed_tasks,
            total_tasks
        )

    with Live(update_progress_table(), refresh_per_second=DEFAULT_REFRESH_RATE, transient=False) as live:
        with rich_joblib_adaptive(job_progress, overall_progress, task_id, overall_task_id, total_cpus):
            results = Parallel(n_jobs=n_jobs, batch_size=batch_size, **kwargs)(
                delayed(f)(i) for i in arr
            )
            # Update completed tasks for final display
            completed_tasks = total_tasks
            live.update(update_progress_table())

    return results

