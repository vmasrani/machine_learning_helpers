from __future__ import annotations

import contextlib
import io
import multiprocessing
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rich.console import Console
from rich.live import Live
from rich.progress import Progress, BarColumn, MofNCompleteColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.text import Text
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
DEFAULT_REFRESH_RATE = 4 * 4

# Thread polling interval for progress updates (seconds)
PROGRESS_POLL_INTERVAL = 0.1


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
def _progress_with_live(desc: str, total: int, disable: bool = False):
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

    # Inject live into callback via closure
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


def _create_queue_sink(log_queue):
    """Create a loguru sink that writes to a multiprocessing queue."""
    def sink(message):
        try:
            log_queue.put_nowait(str(message).rstrip())
        except Exception:
            pass  # Don't let logging errors break the worker
    return sink


def _start_log_consumer(log_queue, live):
    """Start a thread that consumes log messages and prints above progress bar."""
    import queue as queue_module

    stop_event = threading.Event()

    def consumer():
        while not stop_event.is_set():
            try:
                message = log_queue.get(timeout=0.1)
                if message and live:
                    live.console.print(message)
            except queue_module.Empty:
                continue
        # Drain remaining messages
        while True:
            try:
                message = log_queue.get_nowait()
                if message and live:
                    live.console.print(message)
            except queue_module.Empty:
                break

    thread = threading.Thread(target=consumer, daemon=True)
    thread.start()
    return stop_event, thread


def _find_loguru_names(f: Callable) -> set[str]:
    """Find loguru Logger names in function globals without mutating."""
    names: set[str] = set()
    try:
        from loguru._logger import Logger
        if hasattr(f, '__globals__'):
            for name, value in f.__globals__.items():
                if isinstance(value, Logger):
                    names.add(name)
    except ImportError:
        pass
    return names


def _strip_loguru_from_globals(f: Callable, names: set[str]) -> None:
    """Remove loguru Logger from function globals to allow pickling."""
    if not names or not hasattr(f, '__globals__'):
        return
    for name in names:
        if name in f.__globals__:
            f.__globals__[name] = None


@dataclass
class LoguruConfig:
    """Configuration for loguru in worker processes."""
    stripped_names: set[str]
    log_queue: Any  # multiprocessing.Queue


def _setup_worker_loguru(config: LoguruConfig) -> None:
    """Configure loguru in worker process - single place for all loguru setup."""
    if config.log_queue is None:
        return
    try:
        from loguru import logger
        logger.remove()
        logger.add(
            _create_queue_sink(config.log_queue),
            format="{time:HH:mm:ss} | {level: <8} | {message}",
            colorize=False
        )
    except ImportError:
        pass


def _reinject_loguru(f: Callable, stripped_names: set[str]) -> None:
    """Re-inject loguru into function globals after pickle."""
    if not stripped_names or not hasattr(f, '__globals__'):
        return
    try:
        from loguru import logger
        for name in stripped_names:
            if f.__globals__.get(name) is None:
                f.__globals__[name] = logger
    except ImportError:
        pass


def _make_output_capturing_wrapper(f: Callable, config: LoguruConfig) -> Callable:
    """Create wrapper that captures stdout and configures loguru."""
    def wrapper(*args, **kwargs):
        _setup_worker_loguru(config)
        _reinject_loguru(f, config.stripped_names)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = f(*args, **kwargs)
        return (result, buf.getvalue().rstrip())
    return wrapper


@contextlib.contextmanager
def _redirect_loguru_to_live(live):
    """Redirect loguru output to print above the Rich Live display."""
    if live is None:
        yield
        return

    try:
        import loguru
        # Remove existing handlers and add one that prints above progress bar
        loguru.logger.remove()
        handler_id = loguru.logger.add(
            lambda m: live.console.print(m, end=""),
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


@contextlib.contextmanager
def _log_consumer(log_queue: Any, live: Any):
    """Context manager for log consumer thread lifecycle."""
    if log_queue is None or live is None:
        yield
        return

    stop_event, thread = _start_log_consumer(log_queue, live)
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=1.0)


@dataclass
class ParallelMode:
    """Configuration for parallel execution mode."""
    using_threads: bool
    log_queue: Any  # multiprocessing.Queue | None
    wrapped_func: Callable
    stripped_names: set[str]


def _prepare_parallel_mode(f: Callable, prefer: str | None) -> ParallelMode:
    """Prepare function and logging for the selected parallel backend."""
    using_threads = prefer == 'threads'

    if using_threads:
        return ParallelMode(
            using_threads=True,
            log_queue=None,
            wrapped_func=lambda *args, **kw: (f(*args, **kw), ''),
            stripped_names=set()
        )

    stripped_names = _find_loguru_names(f)
    _strip_loguru_from_globals(f, stripped_names)
    log_queue = multiprocessing.Manager().Queue()
    config = LoguruConfig(stripped_names, log_queue)

    return ParallelMode(
        using_threads=False,
        log_queue=log_queue,
        wrapped_func=_make_output_capturing_wrapper(f, config),
        stripped_names=stripped_names
    )


def _print_captured_stdout(results_with_output: list[tuple[Any, str]], using_threads: bool) -> None:
    """Print captured stdout from workers (process mode only)."""
    if using_threads:
        return
    console = Console()
    for _, output in results_with_output:
        if output:
            console.print(output, style="dim cyan")


def _sequential_map(f: Callable, arr: list, desc: str, disable: bool) -> list:
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
        with _redirect_loguru_to_live(live):
            for item in arr:
                results.append(f(item))
                progress.update(task_id, advance=1)
    return results


def pmap(f, arr, n_jobs=-1, disable_tqdm=False, safe_mode=False, spawn=False, batch_size='auto', **kwargs):
    """Parallel map with progress bar.

    Args:
        f: Function to apply to each element
        arr: Iterable of elements to process
        n_jobs: Number of parallel jobs (-1 for all CPUs)
        disable_tqdm: Disable progress bar
        safe_mode: Catch exceptions and return error dicts instead of raising
        spawn: Use spawn multiprocessing start method
        batch_size: Joblib batch size ('auto' or int)
        **kwargs: Additional arguments passed to joblib.Parallel
            - desc: Description for progress bar (default: 'Processing')
            - prefer: 'threads' for threading backend

    Returns:
        List of results from applying f to each element in arr

    Note:
        - stdout from workers is captured and displayed after completion
        - loguru output appears above the progress bar in real-time
    """
    arr = list(arr)
    desc = kwargs.pop('desc', 'Processing')

    if spawn:
        multiprocessing.set_start_method('spawn', force=True)

    f = safe(f) if safe_mode else f

    # Handle n_jobs=1 separately - joblib skips callbacks in sequential mode
    if n_jobs == 1:
        return _sequential_map(f, arr, desc, disable_tqdm)

    mode = _prepare_parallel_mode(f, kwargs.get('prefer'))

    with _progress_with_live(desc, total=len(arr), disable=disable_tqdm) as (progress, live):
        with _log_consumer(mode.log_queue, live):
            if mode.using_threads:
                with _redirect_loguru_to_live(live):
                    results_with_output = Parallel(n_jobs=n_jobs, batch_size=batch_size, **kwargs)(
                        delayed(mode.wrapped_func)(i) for i in arr
                    )
            else:
                results_with_output = Parallel(n_jobs=n_jobs, batch_size=batch_size, **kwargs)(
                    delayed(mode.wrapped_func)(i) for i in arr
                )

    _print_captured_stdout(results_with_output, mode.using_threads)
    return [result for result, _ in results_with_output]


def pmap_df(
    f: Callable,
    df: pd.DataFrame,
    n_chunks: int = 100,
    groups: str | None = None,
    axis: int = 0,
    safe_mode: bool = False,
    **kwargs
) -> pd.DataFrame:
    """Parallel map over DataFrame chunks.

    See: https://towardsdatascience.com/make-your-own-super-pandas-using-multiproc-1c04f41944a1
    """
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


# --- Advanced multi-progress bar implementation (pmap_multi) ---

@contextlib.contextmanager
def rich_joblib_adaptive(job_progress, overall_progress, overall_job_task_id, overall_progress_task_id, total_cpus):
    """Enhanced context manager for joblib with styled progress bars."""
    class RichBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        _job_counter = 0
        _job_counter_lock = threading.Lock()
        _completed_tasks = 0

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

            if self.first_batch and self.batch_size > 0:
                elapsed = current_time - self.start_time
                self.avg_job_time = elapsed / self.batch_size
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
                make_job_description(job_number, total_cpus, self.active_jobs_count + 1),
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

    if spawn:
        multiprocessing.set_start_method('spawn', force=True)

    f = safe(f) if safe_mode else f

    # Handle n_jobs=1 separately - joblib skips callbacks in sequential mode
    if n_jobs == 1:
        return _sequential_map(f, arr, desc, disable_tqdm)

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    total_cpus = min(n_jobs, total_tasks)

    job_progress, overall_progress = create_progress_columns(disable_tqdm)

    task_id = job_progress.add_task(desc, total=total_tasks)
    overall_task_id = overall_progress.add_task(
        Text.assemble(("", "dim blue"), (" Total", "bold white")),
        total=total_tasks
    )

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
            completed_tasks = total_tasks
            live.update(update_progress_table())

    return results
