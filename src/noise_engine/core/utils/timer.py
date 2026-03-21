import time

import torch

# Cache CUDA availability at module level (checked once)
_USE_CUDA = torch.cuda.is_available()


class Timer:
    """
    High-precision execution time measurement utility for CPU and GPU code.

    Ensures accurate GPU timing by synchronizing CUDA operations before
    and after measurement.
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
        """Return elapsed time in seconds."""
        return self.end - self.start
