import time

import torch

# Cache CUDA availability at module level (checked once)
_USE_CUDA = torch.cuda.is_available()


class Timer:
    """High-precision execution time measurement utility for CPU and GPU code.

    Ensures accurate GPU timing by synchronizing CUDA operations before
    and after measurement.

    Example:
        >>> from noise_engine.core.utils.timer import Timer
        >>> with Timer() as t:
        ...     # Some computation here
        ...     result = torch.randn(1000, 1000)
        >>> print(f"Elapsed time: {t.elapsed:.4f} seconds")
        Elapsed time: 0.0012 seconds

    Attributes:
        start (float): Start time in seconds.
        end (float): End time in seconds.
        elapsed (float): Elapsed time in seconds.
    """

    def __enter__(self):
        if _USE_CUDA:
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if _USE_CUDA:
            torch.cuda.synchronize()
        self.end = time.perf_counter()

    @property
    def elapsed(self) -> float:
        """Return elapsed time in seconds.

        Returns:
            float: Elapsed time in seconds.
        """
        return self.end - self.start
