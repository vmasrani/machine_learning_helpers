from __future__ import annotations

import contextlib

import time
import warnings

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    MofNCompleteColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.theme import Theme
from sklearn.model_selection import GroupKFold
# from tqdm.auto import tqdm

# Suppress specific FutureWarning about 'DataFrame.swapaxes'
warnings.filterwarnings(
    "ignore",
    message="'DataFrame.swapaxes' is deprecated and will be removed in a future version. Please use 'DataFrame.transpose' instead."
)


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



def safe(f):
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            error_dict = {'error': str(e), 'args': args, 'kwargs': kwargs}
            print(error_dict)
            return error_dict
    return wrapper




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
