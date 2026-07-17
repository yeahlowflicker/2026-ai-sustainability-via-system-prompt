import time

"""This module provides a warmup function that performs a CPU-intensive computation for a specified duration.
Call the warmup function with the desired duration in seconds to keep the CPU busy and allow it to reach a stable operating temperature."""

def fib(n):
        if n <= 1:
            return n
        return fib(n - 1) + fib(n - 2)

def warmup(duration_seconds: int = 5 * 60):
    t_start = time.time()
    n = 30
    iteration = 0
    # Keep the CPU busy with a pointless but demanding computation until the
    # hardware warms up to its normal operating temperature. Results are discarded.
    while time.time() - t_start < duration_seconds:
        iteration += 1
        elapsed = time.time() - t_start
        print(f'Warmup iteration {iteration}: fib({n}) ({elapsed:.0f}s elapsed)...')
        fib(n)

    total_elapsed = time.time() - t_start
    print(f'Warmup complete: {iteration} iteration(s) over {total_elapsed:.1f}s. Hardware should now be at a stable operating temperature.')