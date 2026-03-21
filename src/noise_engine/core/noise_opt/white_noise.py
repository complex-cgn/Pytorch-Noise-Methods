import torch
from noise_engine.core.device import get_device
from typing import Optional
import logging

class _WhiteNoiseBase:
    """Base class for white noise generators with deterministic behavior."""

    def __init__(self, shape: tuple[int, ...], seed: Optional[int] = None):
        self.shape = tuple(shape)
        self.seed = seed
        self.dim = len(shape)

        if self.dim not in (1, 2, 3):
            raise ValueError(f"Shape must be 1D, 2D, or 3D. Got {self.dim}D")

    def _generate(self) -> torch.Tensor:
        """Generate white noise with deterministic seed handling."""
        device = get_device()

        if self.seed is not None:
            generator = torch.Generator(device=device).manual_seed(self.seed)
            return torch.empty(self.shape, device=device).uniform_(generator=generator)
        else:
            return torch.rand(*self.shape, device=device)

class WhiteNoise1D(_WhiteNoiseBase):
    """Simple white noise generator for 1D data."""

    def __init__(self, shape: tuple[int], seed: Optional[int] = None):
        super().__init__(shape, seed)

    def __call__(self) -> torch.Tensor:
        logging.debug(f"Generating 1D White Noise: shape={self.shape}")
        return self._generate()


class WhiteNoise2D(_WhiteNoiseBase):
    """Simple white noise generator for 2D data."""

    def __init__(self, shape: tuple[int, int], seed: Optional[int] = None):
        super().__init__(shape, seed)

    def __call__(self) -> torch.Tensor:
        logging.debug(f"Generating 2D White Noise: shape={self.shape}")
        return self._generate()


class WhiteNoise3D(_WhiteNoiseBase):
    """Simple white noise generator for 3D data."""

    def __init__(self, shape: tuple[int, int, int], seed: Optional[int] = None):
        super().__init__(shape, seed)

    def __call__(self) -> torch.Tensor:
        logging.debug(f"Generating 3D White Noise: shape={self.shape}")
        return self._generate()