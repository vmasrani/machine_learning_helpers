"""Parallel map with progress bars.

Usage:
    from mlh.parallel import pmap

    results = pmap(fn, items)  # Simple progress bar
    results = pmap(fn, items, show_job_bars=True)  # Per-job progress bars
"""
from __future__ import annotations

import multiprocessing
import time
import warnings
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .loguru_routing import reinject_loguru
from .progress_bars import (
    prepare_parallel_mode,
    sequential_map,
    run_with_simple_bar,
    run_with_job_bars,
)

__all__ = ['pmap', 'pmap_df', 'run_async', 'safe']

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="'DataFrame.swapaxes' is deprecated.*"
)


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


def pmap(f, arr, n_jobs=-1, disable_tqdm=False, safe_mode=False, spawn=False, batch_size='auto', show_job_bars=False, **kwargs):
    """Parallel map with progress bar.

    Args:
        f: Function to apply to each element
        arr: Iterable of elements to process
        n_jobs: Number of parallel jobs (-1 for all CPUs)
        disable_tqdm: Disable progress bar
        safe_mode: Catch exceptions and return error dicts instead of raising
        spawn: Use spawn multiprocessing start method
        batch_size: Joblib batch size ('auto' or int)
        show_job_bars: Show per-job progress bars with CPU info (default: False)
        **kwargs: Additional arguments passed to joblib.Parallel
            - desc: Description for progress bar (default: 'Processing')
            - prefer: 'threads' for threading backend

    Returns:
        List of results from applying f to each element in arr

    Note:
        - In process mode, all output (print, rich.print, loguru) is routed through
          loguru and appears above the progress bar in real-time
        - In thread mode, output goes directly to stdout (shared between threads)
    """
    arr = list(arr)
    desc = kwargs.pop('desc', 'Processing')
    total_tasks = len(arr)

    if spawn:
        multiprocessing.set_start_method('spawn', force=True)

    f = safe(f) if safe_mode else f

    if n_jobs == 1:
        return sequential_map(f, arr, desc, disable_tqdm)

    mode = prepare_parallel_mode(f, kwargs.get('prefer'))

    actual_n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()
    total_cpus = min(actual_n_jobs, total_tasks)

    try:
        if show_job_bars:
            results = run_with_job_bars(
                mode, arr, n_jobs, batch_size, disable_tqdm, desc, total_tasks, total_cpus, kwargs
            )
        else:
            results = run_with_simple_bar(
                mode, arr, n_jobs, batch_size, disable_tqdm, desc, kwargs
            )
    finally:
        if mode.stripped_names:
            reinject_loguru(f, mode.stripped_names)

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
    """Parallel map over DataFrame chunks.

    See: https://towardsdatascience.com/make-your-own-super-pandas-using-multiproc-1c04f41944a1
    """
    if groups:
        n_chunks = min(n_chunks, df[groups].nunique())
        group_kfold = GroupKFold(n_splits=n_chunks)
        df_split = [df.iloc[test_index] for _, test_index in group_kfold.split(df, groups=df[groups])]
    else:
        df_split = np.array_split(df, n_chunks)
    df = pd.concat(pmap(f, df_split, safe_mode=safe_mode, **kwargs), axis=axis)
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
