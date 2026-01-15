"""Re-export from standalone pmap package for backward compatibility.

Usage:
    from mlh.parallel import pmap

    results = pmap(fn, items)  # Simple progress bar
    results = pmap(fn, items, show_job_bars=True)  # Per-job progress bars

Note: This module re-exports from the standalone 'pmap' package.
For new projects, prefer importing directly: from pmap import pmap
"""
from pmap import pmap, pmap_df, run_async, safe

__all__ = ['pmap', 'pmap_df', 'run_async', 'safe']
