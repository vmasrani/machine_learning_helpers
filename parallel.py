from __future__ import annotations

import contextlib
import multiprocessing
import threading
import time
import warnings
from typing import Any, Callable, Iterable

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rich.console import Console
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

from progress_styles import (
    create_progress_columns,
    create_progress_table,
    make_job_description,
)

# from tqdm.auto import tqdm

# Suppress specific FutureWarning about 'DataFrame.swapaxes'
warnings.filterwarnings(
    "ignore",
    message="'DataFrame.swapaxes' is deprecated and will be removed in a future version. Please use 'DataFrame.transpose' instead."
)


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
                time.sleep(0.1)

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

    joblib.parallel.BatchCompletionCallBack = WrappedCallback
    try:
        yield
    finally:
        if current_callback_instance is not None:
            current_callback_instance.stop()
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        job_progress.update(overall_job_task_id, completed=job_progress.tasks[overall_job_task_id].total)
        overall_progress.update(overall_progress_task_id, completed=overall_progress.tasks[overall_progress_task_id].total)


def safe(f: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            error_dict = {'error': str(e), 'args': args, 'kwargs': kwargs}
            print(error_dict)
            return error_dict
    return wrapper


def pmap_multi(
    f: Callable,
    arr: Iterable[Any],
    n_jobs: int = -1,
    disable_tqdm: bool = False,
    safe_mode: bool = False,
    spawn: bool = False,
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

    with Live(update_progress_table(), refresh_per_second=10, transient=False) as live:
        with rich_joblib_adaptive(job_progress, overall_progress, task_id, overall_task_id, total_cpus):
            results = Parallel(n_jobs=n_jobs, **kwargs)(
                delayed(f)(i) for i in arr
            )
            # Update completed tasks for final display
            completed_tasks = total_tasks
            live.update(update_progress_table())

    return results





@contextlib.contextmanager
def rich_joblib(progress, task_id):
    """Context manager to patch joblib to report into rich progress bar"""
    class RichBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        def __call__(self, *args, **kwargs):
            progress.update(task_id, advance=self.batch_size)
            return super().__call__(*args, **kwargs)
    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = RichBatchCompletionCallback
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        progress.update(task_id, completed=progress.tasks[task_id].total)
        progress.refresh()
        progress.update(task_id, visible=False)


def pmap(f, arr, n_jobs=-1, disable_tqdm=False, safe_mode=False, spawn=False, **kwargs):
    arr = list(arr)  # convert generators to list so progress works
    desc = kwargs.pop('desc', 'Processing')

    if spawn:
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)

    # Add this configuration
    # kwargs['mmap_mode'] = 'r+'  # Enable memory mapping for better performance

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

    f = safe(f) if safe_mode else f

    with progress_bar as progress:
        task_id = progress.add_task(desc, total=len(arr))
        with rich_joblib(progress, task_id):
            return Parallel(n_jobs=n_jobs, **kwargs)(delayed(f)(i) for i in arr)



# @contextlib.contextmanager
# def tqdm_joblib(tqdm_object):
#     # from https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/49950707
#     """Context manager to patch joblib to report into tqdm progress bar given as argument"""
#     class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
#         def __init__(self, *args, **kwargs):
#             super().__init__(*args, **kwargs)

#         def __call__(self, *args, **kwargs):
#             tqdm_object.update(n=self.batch_size)
#             return super().__call__(*args, **kwargs)

#     old_batch_callback = joblib.parallel.BatchCompletionCallBack
#     joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
#     try:
#         yield tqdm_object
#     finally:
#         joblib.parallel.BatchCompletionCallBack = old_batch_callback
#         tqdm_object.close()



# def pmap_old(f, arr, n_jobs=-1, disable_tqdm=False, safe_mode=False, **kwargs):
#     arr = list(arr)  # convert generators to list so tqdm works
#     desc = kwargs.pop('desc', None)
#     f = safe(f) if safe_mode else f
#     with tqdm_joblib(tqdm(total=len(arr), disable=disable_tqdm, desc=desc)) as progress_bar:
#         return Parallel(n_jobs=n_jobs, **kwargs)(delayed(f)(i) for i in arr)


def pmap_df(f, df, n_chunks=100, groups=None, axis=0, safe_mode=False, **kwargs):
    # https://towardsdatascience.com/make-your-own-super-pandas-using-multiproc-1c04f41944a1
    if groups:
        n_chunks = min(n_chunks, df[groups].nunique())
        group_kfold = GroupKFold(n_splits=n_chunks)
        df_split = [df.iloc[test_index] for _, test_index in group_kfold.split(df, groups=df[groups])]
    else:
        df_split = np.array_split(df, n_chunks)
    df = pd.concat(pmap(f, df_split, safe_mode=safe_mode, **kwargs), axis=axis)
    return df


def run_async(func):
    """
    # example
    # @run_async
    # def long_run(idx, val='cat'):
    #     for i in range(idx):
    #         print(i)
    #         time.sleep(1)
    #     return val

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
