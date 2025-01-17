from functools import wraps
import time
import psutil
from typing import Iterable, Callable, TypeVar, Iterator
from tqdm import tqdm
from statistics import mean
from collections import deque

T = TypeVar('T')
R = TypeVar('R')

class ProcessMonitor:
    def __init__(self, total: int):
        self.total = total
        self.processed = 0
        self.start_time = time.time()
        self.processing_times = deque(maxlen=50)  # Rolling window of last 50 items
        self.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

    def update(self) -> dict:
        """Calculate and return current monitoring stats."""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        elapsed = time.time() - self.start_time
        cpu_percent = psutil.Process().cpu_percent()

        stats = {
            'memory': f'{current_memory:.1f}MB',
            'mem_delta': f'{(current_memory - self.initial_memory):+.1f}MB',
            'processed': f'{self.processed}/{self.total}',
            'remaining': self.total - self.processed,
            'elapsed': f'{elapsed/60:.1f}min',
            'items/sec': f'{self.processed/elapsed:.1f}',
            'cpu%': f'{cpu_percent:.1f}',
            'threads': len(psutil.Process().threads())
        }

        if self.processing_times:
            avg_time = mean(self.processing_times)
            est_remaining = avg_time * (self.total - self.processed)
            min_time = min(self.processing_times)
            max_time = max(self.processing_times)
            stats.update({
                'avg_time': f'{avg_time:.2f}s',
                'min_time': f'{min_time:.2f}s',
                'max_time': f'{max_time:.2f}s',
                'est_remaining': f'{est_remaining/60:.1f}min'
            })

        return stats

def tmap(
    func: Callable[[T], R],
    iterable: Iterable[T],
    desc: str = "Processing",
    monitor_mem: bool = True,
    ignore_errors: bool = False,
    position: int = 0,
    **tqdm_kwargs
) -> Iterator[R]:
    """
    Monitored mapping function with progress bar and performance metrics.

    Args:
        func: Function to apply to each item
        iterable: Input iterable
        desc: Description for the progress bar
        monitor_mem: Whether to track memory usage
        ignore_errors: Whether to continue on errors or raise them
        **tqdm_kwargs: Additional arguments to pass to tqdm

    Returns:
        Iterator of results

    Example:
        >>> from monitor import tmap
        >>> results = list(tmap(process_file, files))
    """
    # Attempt to get total length if possible
    try:
        total = len(iterable)
    except TypeError:
        total = None

    # Create the monitor object if memory monitoring is enabled
    monitor = ProcessMonitor(total=total or 0) if monitor_mem else None

    # Use a direct list if length is known, otherwise iterate directly.
    if total is not None:
        iterator = list(iterable)
    else:
        iterator = iterable

    tqdm_kwargs.update({
        'position': position,
        'leave': True,
        'dynamic_ncols': True
    })

    with tqdm(iterator, desc=desc, total=total, **tqdm_kwargs) as pbar:
        # Create one line for stats that we'll update
        if monitor_mem:
            print("")  # Single blank line for stats

        for item in pbar:
            start_time = time.time()

            try:
                result = func(item)
            except Exception as e:
                if ignore_errors:
                    print(f"Error processing item {item}: {e}")
                    continue
                else:
                    raise

            if monitor:
                monitor.processing_times.append(time.time() - start_time)
                monitor.processed += 1
                stats = monitor.update()
                stats_str = ' | '.join(f"{k}: {v}" for k, v in stats.items())
                # Clear the line and move cursor up to print stats
                print(f"\033[F\033[K{stats_str}")

            yield result


#


