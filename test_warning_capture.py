import time
import warnings
from mlh.parallel import pmap

def process_with_warnings(i):
    """Function that generates warnings and prints."""
    print(f"Processing item {i}")
    if i % 3 == 0:
        warnings.warn(f"Warning for item {i}: This is a test warning", UserWarning)
    time.sleep(0.01)
    return i * 2

if __name__ == "__main__":
    print("Testing warning capture with pmap:")
    print("="*60)
    results = pmap(process_with_warnings, range(20), n_jobs=4, batch_size=1)
    print("="*60)
    print(f"\nCompleted processing {len(results)} items")
    print("All warnings should have appeared above the progress bar!")
