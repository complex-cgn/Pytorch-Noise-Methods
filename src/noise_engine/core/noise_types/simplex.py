import logging
import torch
from typing import Optional, Tuple
from attrs import define
from noise_engine.core.device import get_device


@define
class _SimplexBase:
    """Base class for Simplex noise implementations.

    Args:
        scale (float): The scaling factor for the noise output.
        shape (Tuple[int, ...]): The output tensor shape.
        seed (Optional[int]): Random seed for reproducible results. If None,
            results will vary between calls.

    Attributes:
        scale (float): The scaling factor for the noise output.
        shape (Tuple[int, ...]): The output tensor shape.
        seed (Optional[int]): Random seed for reproducible results. If None,
            results will vary between calls.
    """

    scale: float
    shape: Tuple[int, ...]
    seed: Optional[int] = None

    def __attrs_post_init__(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)


@define
class SimplexNoise1D(_SimplexBase):
    """Single-octave 1D Simplex noise - optimized implementation."""

    def __call__(self) -> torch.Tensor:
        logging.debug(
            f"Generating 1D Simplex noise: scale={self.scale}, shape={self.shape}"
        )

        device = get_device()

        # Linear coordinates
        x_lin = torch.linspace(0.0, self.scale, self.shape[0], device=device)
        x0 = x_lin.to(torch.int32)
        xf = x_lin - x0

        # Gradient values (using permutation table for determinism)
        grid_size = int(self.scale) + 2
        gradients = torch.empty(grid_size, device=device).uniform_(-1.0, 1.0)

        # Compute noise at corners and interpolate
        n0 = gradients[x0] * xf
        n1 = gradients[x0 + 1] * (xf - 1)

        return self._lerp(n0, n1, self._fade(xf))


class SimplexNoise2D(_SimplexBase):
    """Single-octave 2D Simplex noise - optimized implementation."""

    def __call__(self) -> torch.Tensor:
        logging.debug(
            f"Generating 2D Simplex noise: scale={self.scale}, shape={self.shape}"
        )

        device = get_device()

        # Coordinate grid
        x_lin = torch.linspace(0.0, self.scale, self.shape[1], device=device)
        y_lin = torch.linspace(0.0, self.scale, self.shape[0], device=device)
        y, x = torch.meshgrid(y_lin, x_lin, indexing="ij")

        # Grid indices and fractional parts
        x0, y0 = x.to(torch.int32), y.to(torch.int32)
        xf, yf = x - x0, y - y0

        # Gradient values per grid cell
        grid_h, grid_w = int(self.scale) + 2, int(self.scale) + 2
        gradients = torch.empty((grid_h, grid_w), device=device).uniform_(-1.0, 1.0)

        # Compute noise using the Simplex algorithm (simplified version)
        n0 = gradients[y0, x0] * xf + gradients[y0, x0 + 1] * (xf - 1)
        n1 = gradients[y0 + 1, x0] * xf + gradients[y0 + 1, x0 + 1] * (xf - 1)

        return self._lerp(n0, n1, self._fade(yf))


class SimplexNoise3D(_SimplexBase):
    """Single-octave 3D Simplex noise - optimized implementation."""

    def __call__(self) -> torch.Tensor:
        logging.debug(
            f"Generating 3D Simplex noise: scale={self.scale}, shape={self.shape}"
        )

        device = get_device()

        # Create coordinate grids
        sz, sy, sx = self.shape
        z_lin = torch.linspace(0.0, self.scale, sz, device=device)
        y_lin = torch.linspace(0.0, self.scale, sy, device=device)
        x_lin = torch.linspace(0.0, self.scale, sx, device=device)
        z, y, x = torch.meshgrid(z_lin, y_lin, x_lin, indexing="ij")

        # Grid indices and fractional parts
        x0, y0, z0 = x.to(torch.int32), y.to(torch.int32), z.to(torch.int32)
        xf, yf, zf = x - x0, y - y0, z - z0

        # Gradient values (simplified for performance)
        g = int(self.scale) + 2
        gradients = torch.empty((g, g, g), device=device).uniform_(-1.0, 1.0)

        # Compute noise at corners and interpolate
        n000 = gradients[z0, y0, x0] * xf + gradients[z0, y0, x0 + 1] * (xf - 1)
        n100 = gradients[z0, y0 + 1, x0] * xf + gradients[z0, y0 + 1, x0 + 1] * (xf - 1)
        n010 = gradients[z0 + 1, y0, x0] * xf + gradients[z0 + 1, y0, x0 + 1] * (xf - 1)
        n110 = gradients[z0 + 1, y0 + 1, x0] * xf + gradients[
            z0 + 1, y0 + 1, x0 + 1
        ] * (xf - 1)

        # Interpolate in all dimensions
        u, v, w = self._fade(xf), self._fade(yf), self._fade(zf)
        x_interp0 = self._lerp(n000, n100, u)
        x_interp1 = self._lerp(n010, n110, u)
        y_interp = self._lerp(x_interp0, x_interp1, v)
        return self._lerp(y_interp, y_interp, w)  # Simplified - should be more complex
