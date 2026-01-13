#!/usr/bin/env -S uv run
import time
from loguru import logger
from mlh.parallel import pmap
import rich

def process_with_loguru(i):
    """Function that uses loguru for logging."""
    logger.info(f"Processing item {i}")
    if i % 3 == 0:
        logger.warning(f"Item {i} is divisible by 3")
    time.sleep(4)
    return i * 2

if __name__ == "__main__":
    print("Testing loguru capture with pmap:")
    print("="*60)
    # Test processes mode (default) - now works with queue-based log forwarding
    # results = pmap(process_with_loguru, range(10), n_jobs=2)
    # results = pmap(process_with_loguru, range(10), prefer='threads', n_jobs=2)


    results = pmap(process_with_loguru, range(10), n_jobs=4, show_job_bars=True)  # Process mode
    results = pmap(process_with_loguru, range(10), prefer='threads', n_jobs=4, show_job_bars=True)  # Thread mode


    print("="*60)
    print(f"\nCompleted processing {len(results)} items")
    print(f"Sample results: {results[:5]}...")
    print("All loguru output should have appeared above the progress bar!")
