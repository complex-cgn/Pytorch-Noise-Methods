import logging
import torch
from typing import Optional, Tuple
from attrs import define
from noise_engine.core.device import get_device


@define
class _WhiteBase:
    """Base class for White noise implementations.

    Args:
        shape (Tuple[int, ...]): The output tensor shape.
        seed (Optional[int]): Random seed for reproducible results. If None,
            results will vary between calls.

    Attributes:
        shape (Tuple[int, ...]): The output tensor shape.
        seed (Optional[int]): Random seed for reproducible results. If None,
            results will vary between calls.
    """

    shape: Tuple[int, ...]
    seed: Optional[int] = None

    def __attrs_post_init__(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)


@define
class WhiteNoise1D(_WhiteBase):
    """Single-octave 1D White noise - optimized implementation.

    Returns:
        torch.Tensor: A 1D tensor of shape specified by the 'shape' parameter,
            containing uniform random noise values in the range [0, 1].

    Example:
        >>> noise_gen = WhiteNoise1D(shape=(100,))
        >>> noise = noise_gen()
        >>> print(noise.shape)
        torch.Size([100])
    """

    def __call__(self) -> torch.Tensor:
        logging.debug(
            f"Generating 1D White noise: shape={self.shape}"
        )

        device = get_device()

        # Generate uniform random noise
        return torch.rand(self.shape, device=device)


class WhiteNoise2D(_WhiteBase):
    """Single-octave 2D White noise - optimized implementation.

    Returns:
        torch.Tensor: A 2D tensor of shape specified by the 'shape' parameter,
            containing uniform random noise values in the range [0, 1].

    Example:
        >>> noise_gen = WhiteNoise2D(shape=(100, 100))
        >>> noise = noise_gen()
        >>> print(noise.shape)
        torch.Size([100, 100])
    """

    def __call__(self) -> torch.Tensor:
        logging.debug(
            f"Generating 2D White noise: shape={self.shape}"
        )

        device = get_device()

        # Generate uniform random noise
        return torch.rand(self.shape, device=device)


class WhiteNoise3D(_WhiteBase):
    """Single-octave 3D White noise - optimized implementation.

    Returns:
        torch.Tensor: A 3D tensor of shape specified by the 'shape' parameter,
            containing uniform random noise values in the range [0, 1].

    Example:
        >>> noise_gen = WhiteNoise3D(shape=(50, 50, 50))
        >>> noise = noise_gen()
        >>> print(noise.shape)
        torch.Size([50, 50, 50])
    """

    def __call__(self) -> torch.Tensor:
        logging.debug(
            f"Generating 3D White noise: shape={self.shape}"
        )

        device = get_device()

        # Generate uniform random noise
        return torch.rand(self.shape, device=device)
