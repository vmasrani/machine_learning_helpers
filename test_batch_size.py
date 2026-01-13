import time
from mlh.parallel import pmap

def process_item(i):
    """Simulate some work with occasional prints."""
    print(f"Processing item {i}")
    time.sleep(0.01)
    if i % 3 == 0:
        print(f"  -> Item {i} is divisible by 3!")
    return i * 2

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing with batch_size=1 (smooth progress)")
    print("="*60)
    start = time.time()
    results = pmap(process_item, range(100), n_jobs=8, batch_size=1)
    elapsed_1 = time.time() - start
    print(f"\nCompleted in {elapsed_1:.2f}s with batch_size=1")

    print("\n" + "="*60)
    print("Testing with batch_size='auto' (adaptive batching)")
    print("="*60)
    start = time.time()
    results = pmap(process_item, range(100), n_jobs=8, batch_size='auto')
    elapsed_auto = time.time() - start
    print(f"\nCompleted in {elapsed_auto:.2f}s with batch_size='auto'")

    print("\n" + "="*60)
    print(f"Performance difference: {((elapsed_1 - elapsed_auto) / elapsed_auto * 100):.1f}%")
    print("="*60)
