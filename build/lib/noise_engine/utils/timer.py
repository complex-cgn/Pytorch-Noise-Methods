import time

import torch


class Timer:
    """
    High-precision execution time measurement utility for CPU and GPU code.

    Ensures accurate GPU timing by synchronizing CUDA operations before
    and after measurement.
    """

    def __enter__(self):

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end = time.perf_counter()

    @property
    def elapsed(self):
        return self.end - self.start
