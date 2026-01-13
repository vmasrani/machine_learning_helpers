#!/usr/bin/env -S uv run
import time
from loguru import logger
from mlh.parallel import pmap
import rich

def process_with_loguru(i):
    """Function that uses loguru for logging."""
    # rich.print(f"Processing item {i}")
    logger.info(f"Processing item {i}")
    # print(f"Processing item {i}")
    if i % 3 == 0:
        logger.warning(f"Item {i} is divisible by 3")
    time.sleep(1)
    return i * 2

if __name__ == "__main__":
    print("Testing loguru capture with pmap:")
    print("="*60)
    # Test processes mode (default) - now works with queue-based log forwarding
    # results = pmap(process_with_loguru, range(10), n_jobs=2)
    # results = pmap(process_with_loguru, range(10), prefer='threads', n_jobs=2)


    results = pmap(process_with_loguru, range(10), n_jobs=1) # FAILS
    results = pmap(process_with_loguru, range(10), prefer='threads', n_jobs=1) # WORKS BUT PRINTS ARE WEIRD AND DONT DISPLAY NICELY ABOVE THE PROGRESS BAR


    print("="*60)
    print(f"\nCompleted processing {len(results)} items")
    print(f"Sample results: {results[:5]}...")
    print("All loguru output should have appeared above the progress bar!")
