"""
Concurrency Patterns

Features:
- Thread pool management
- Process pool management
- Async/await patterns
- Queue-based pipelines
- Rate limiting
- Resource pooling
- Progress tracking
"""

from typing import Callable, List, Any, Optional, Iterator
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import threading
import multiprocessing
import time
from dataclasses import dataclass
import asyncio


@dataclass
class TaskResult:
    """Result from async task."""
    index: int
    result: Any
    error: Optional[Exception] = None
    duration: float = 0.0


class ThreadPool:
    """Thread pool for concurrent execution."""

    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize thread pool.

        Parameters
        ----------
        max_workers : int, optional
            Maximum number of worker threads
        """
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def map(
        self,
        func: Callable,
        items: List[Any],
        show_progress: bool = False
    ) -> List[Any]:
        """
        Map function over items in parallel.

        Parameters
        ----------
        func : callable
            Function to apply
        items : list
            Input items
        show_progress : bool, default=False
            Show progress bar

        Returns
        -------
        results : list
            Results in order
        """
        futures = {self.executor.submit(func, item): i for i, item in enumerate(items)}
        results = [None] * len(items)

        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = e

            completed += 1
            if show_progress:
                progress = completed / len(items) * 100
                print(f"\rProgress: {progress:.1f}%", end='', flush=True)

        if show_progress:
            print()  # New line

        return results

    def submit(self, func: Callable, *args, **kwargs):
        """
        Submit single task.

        Parameters
        ----------
        func : callable
            Function to execute
        *args, **kwargs
            Function arguments

        Returns
        -------
        future : Future
            Future object
        """
        return self.executor.submit(func, *args, **kwargs)

    def shutdown(self, wait: bool = True):
        """Shutdown thread pool."""
        self.executor.shutdown(wait=wait)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


class ProcessPool:
    """Process pool for CPU-intensive tasks."""

    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize process pool.

        Parameters
        ----------
        max_workers : int, optional
            Maximum number of worker processes
        """
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)

    def map(
        self,
        func: Callable,
        items: List[Any],
        chunksize: Optional[int] = None,
        show_progress: bool = False
    ) -> List[Any]:
        """
        Map function over items using processes.

        Parameters
        ----------
        func : callable
            Function to apply
        items : list
            Input items
        chunksize : int, optional
            Chunk size for batching
        show_progress : bool, default=False
            Show progress

        Returns
        -------
        results : list
            Results in order
        """
        if chunksize is None:
            chunksize = max(1, len(items) // (self.max_workers * 4))

        futures = {self.executor.submit(func, item): i for i, item in enumerate(items)}
        results = [None] * len(items)

        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = e

            completed += 1
            if show_progress:
                progress = completed / len(items) * 100
                print(f"\rProgress: {progress:.1f}%", end='', flush=True)

        if show_progress:
            print()

        return results

    def shutdown(self, wait: bool = True):
        """Shutdown process pool."""
        self.executor.shutdown(wait=wait)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


class Pipeline:
    """Queue-based processing pipeline."""

    def __init__(
        self,
        stages: List[Callable],
        queue_size: int = 100,
        num_workers: int = 1
    ):
        """
        Initialize pipeline.

        Parameters
        ----------
        stages : list
            List of processing stage functions
        queue_size : int, default=100
            Size of intermediate queues
        num_workers : int, default=1
            Number of workers per stage
        """
        self.stages = stages
        self.queue_size = queue_size
        self.num_workers = num_workers
        self.queues = [queue.Queue(maxsize=queue_size) for _ in range(len(stages) + 1)]
        self.workers = []
        self.stop_event = threading.Event()

    def _worker(self, stage_fn: Callable, input_queue: queue.Queue, output_queue: queue.Queue):
        """Worker thread for pipeline stage."""
        while not self.stop_event.is_set():
            try:
                item = input_queue.get(timeout=0.1)
                if item is None:  # Sentinel
                    break

                # Process
                result = stage_fn(item)

                # Pass to next stage
                output_queue.put(result)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}")

    def start(self):
        """Start pipeline workers."""
        for i, stage_fn in enumerate(self.stages):
            for _ in range(self.num_workers):
                worker = threading.Thread(
                    target=self._worker,
                    args=(stage_fn, self.queues[i], self.queues[i + 1]),
                    daemon=True
                )
                worker.start()
                self.workers.append(worker)

    def process(self, items: List[Any]) -> Iterator[Any]:
        """
        Process items through pipeline.

        Parameters
        ----------
        items : list
            Input items

        Yields
        ------
        result : Any
            Processed results
        """
        self.start()

        # Feed items to first queue
        for item in items:
            self.queues[0].put(item)

        # Sentinel values
        for _ in range(self.num_workers):
            self.queues[0].put(None)

        # Collect results from last queue
        num_results = 0
        while num_results < len(items):
            try:
                result = self.queues[-1].get(timeout=1.0)
                if result is not None:
                    yield result
                    num_results += 1
            except queue.Empty:
                continue

    def stop(self):
        """Stop pipeline."""
        self.stop_event.set()
        for worker in self.workers:
            worker.join(timeout=1.0)


class RateLimiter:
    """Rate limiter for API calls or resource access."""

    def __init__(self, max_calls: int, time_window: float = 1.0):
        """
        Initialize rate limiter.

        Parameters
        ----------
        max_calls : int
            Maximum calls per time window
        time_window : float, default=1.0
            Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self.lock = threading.Lock()

    def acquire(self) -> bool:
        """
        Acquire permission to make a call.

        Returns
        -------
        allowed : bool
            Whether call is allowed
        """
        with self.lock:
            now = time.time()

            # Remove old calls outside time window
            self.calls = [t for t in self.calls if now - t < self.time_window]

            # Check if we can make a call
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True
            else:
                # Wait time until next available slot
                oldest = self.calls[0]
                wait_time = self.time_window - (now - oldest)
                time.sleep(wait_time)

                # Retry
                return self.acquire()

    def __call__(self, func: Callable) -> Callable:
        """Decorator for rate-limited function."""
        def wrapper(*args, **kwargs):
            self.acquire()
            return func(*args, **kwargs)
        return wrapper


class ResourcePool:
    """Pool of reusable resources."""

    def __init__(self, factory: Callable, max_size: int = 10):
        """
        Initialize resource pool.

        Parameters
        ----------
        factory : callable
            Function to create new resources
        max_size : int, default=10
            Maximum pool size
        """
        self.factory = factory
        self.max_size = max_size
        self.pool = queue.Queue(maxsize=max_size)
        self.size = 0
        self.lock = threading.Lock()

    def acquire(self):
        """
        Acquire resource from pool.

        Returns
        -------
        resource : Any
            Resource object
        """
        try:
            return self.pool.get_nowait()
        except queue.Empty:
            with self.lock:
                if self.size < self.max_size:
                    self.size += 1
                    return self.factory()
                else:
                    return self.pool.get()

    def release(self, resource: Any):
        """
        Release resource back to pool.

        Parameters
        ----------
        resource : Any
            Resource to release
        """
        try:
            self.pool.put_nowait(resource)
        except queue.Full:
            # Pool is full, discard resource
            pass

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass  # Context manager protocol for resources


class AsyncTaskManager:
    """Async task manager for coroutines."""

    def __init__(self):
        """Initialize async task manager."""
        self.tasks = []

    async def run_task(
        self,
        coro: Any,
        index: int
    ) -> TaskResult:
        """
        Run single async task.

        Parameters
        ----------
        coro : coroutine
            Coroutine to execute
        index : int
            Task index

        Returns
        -------
        result : TaskResult
            Task result
        """
        start_time = time.time()
        try:
            result = await coro
            duration = time.time() - start_time
            return TaskResult(index=index, result=result, duration=duration)
        except Exception as e:
            duration = time.time() - start_time
            return TaskResult(index=index, result=None, error=e, duration=duration)

    async def run_all(
        self,
        coros: List[Any],
        max_concurrent: Optional[int] = None
    ) -> List[TaskResult]:
        """
        Run multiple coroutines concurrently.

        Parameters
        ----------
        coros : list
            List of coroutines
        max_concurrent : int, optional
            Maximum concurrent tasks

        Returns
        -------
        results : list
            Task results in order
        """
        if max_concurrent:
            semaphore = asyncio.Semaphore(max_concurrent)

            async def limited_task(coro, idx):
                async with semaphore:
                    return await self.run_task(coro, idx)

            tasks = [limited_task(coro, i) for i, coro in enumerate(coros)]
        else:
            tasks = [self.run_task(coro, i) for i, coro in enumerate(coros)]

        results = await asyncio.gather(*tasks)
        return results


class ProgressTracker:
    """Track progress of concurrent tasks."""

    def __init__(self, total: int):
        """
        Initialize progress tracker.

        Parameters
        ----------
        total : int
            Total number of tasks
        """
        self.total = total
        self.completed = 0
        self.lock = threading.Lock()
        self.start_time = time.time()

    def update(self, n: int = 1):
        """
        Update progress.

        Parameters
        ----------
        n : int, default=1
            Number of tasks completed
        """
        with self.lock:
            self.completed += n
            self._print_progress()

    def _print_progress(self):
        """Print progress bar."""
        elapsed = time.time() - self.start_time
        progress = self.completed / self.total
        bar_length = 40
        filled = int(bar_length * progress)
        bar = '=' * filled + '-' * (bar_length - filled)

        # Estimate remaining time
        if self.completed > 0:
            eta = elapsed / self.completed * (self.total - self.completed)
            eta_str = f"{eta:.1f}s"
        else:
            eta_str = "?"

        print(f"\r[{bar}] {self.completed}/{self.total} ({progress*100:.1f}%) ETA: {eta_str}",
              end='', flush=True)

        if self.completed >= self.total:
            print()  # New line


# Convenience functions
def parallel_map(
    func: Callable,
    items: List[Any],
    use_processes: bool = False,
    max_workers: Optional[int] = None,
    show_progress: bool = False
) -> List[Any]:
    """
    Parallel map function.

    Parameters
    ----------
    func : callable
        Function to apply
    items : list
        Input items
    use_processes : bool, default=False
        Use processes instead of threads
    max_workers : int, optional
        Maximum workers
    show_progress : bool, default=False
        Show progress

    Returns
    -------
    results : list
        Results in order
    """
    if use_processes:
        with ProcessPool(max_workers) as pool:
            return pool.map(func, items, show_progress=show_progress)
    else:
        with ThreadPool(max_workers) as pool:
            return pool.map(func, items, show_progress=show_progress)


def rate_limited(max_calls: int, time_window: float = 1.0):
    """
    Decorator for rate-limited function.

    Parameters
    ----------
    max_calls : int
        Maximum calls per time window
    time_window : float, default=1.0
        Time window in seconds

    Returns
    -------
    decorator : callable
        Rate-limiting decorator
    """
    limiter = RateLimiter(max_calls, time_window)
    return limiter
