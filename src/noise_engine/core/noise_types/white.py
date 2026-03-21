import logging
import torch
from typing import Optional, Tuple
from attrs import define
from noise_engine.core.device import get_device


@define
class _WhiteBase:
    """Base class for Simplex noise implementations."""

    shape: Tuple[int, ...]
    seed: Optional[int] = None

    def __attrs_post_init__(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)


@define
class WhiteNoise1D(_WhiteBase):
    """Single-octave 1D White noise - optimized implementation."""

    def __call__(self) -> torch.Tensor:
        logging.debug(
            f"Generating 1D White noise: scale={self.scale}, shape={self.shape}"
        )

        device = get_device()

        # Generate uniform random noise
        return torch.rand(self.shape, device=device)


class WhiteNoise2D(_WhiteBase):
    """Single-octave 2D White noise - optimized implementation."""

    def __call__(self) -> torch.Tensor:
        logging.debug(
            f"Generating 2D White noise: scale={self.scale}, shape={self.shape}"
        )

        device = get_device()

        # Generate uniform random noise
        return torch.rand(self.shape, device=device)


class WhiteNoise3D(_WhiteBase):
    """Single-octave 3D White noise - optimized implementation."""

    def __call__(self) -> torch.Tensor:
        logging.debug(
            f"Generating 3D White noise: scale={self.scale}, shape={self.shape}"
        )

        device = get_device()

        # Generate uniform random noise
        return torch.rand(self.shape, device=device)
