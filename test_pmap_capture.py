#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "joblib",
#     "rich",
#     "numpy",
#     "pandas",
#     "scikit-learn",
# ]
# ///

import time
from mlh.parallel import pmap

def process_with_prints(i):
    """Function with print statements that would normally interfere with progress bar."""
    print(f"Processing item {i}")
    time.sleep(0.15)
    if i % 3 == 0:
        print(f"  -> Item {i} is divisible by 3!")
    return i * 2

if __name__ == "__main__":
    print("Testing pmap with captured print statements:\n")
    results = pmap(process_with_prints, range(1000), n_jobs=8)
    print(f"\nResults: {results}")
