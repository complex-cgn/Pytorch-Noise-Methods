import logging
from typing import Optional, Tuple
import torch
from attrs import define
from noise_engine.core.device import get_device


@define
class _PerlinBase:
    """Base class for Perlin noise with shared functionality."""

    scale: float
    shape: Tuple[int, ...]
    seed: Optional[int] = None

    def __attrs_post_init__(self):
        if len(self.shape) not in (1, 2, 3):
            raise ValueError(f"Shape must be 1D, 2D, or 3D. Got {len(self.shape)}D")
        if self.scale < 0:
            raise ValueError("Scale must be positive.")
        if self.seed is not None:
            torch.manual_seed(self.seed)

    @staticmethod
    def _fade(t: torch.Tensor) -> torch.Tensor:
        """
        Compute the quintic fade curve for smooth interpolation.

        Uses Horner's method: t³ * (t² * 6 - 15*t + 10)
        Equivalent to: 6t⁵ - 15t⁴ + 10t³

        Args:
            t: Input tensor in range [0, 1]

        Returns:
            Smoothed tensor in range [0, 1] with zero first and second derivatives at endpoints
        """
        return t * t * (t * (t * (t * 6.0 - 15.0) + 10.0))

    @staticmethod
    def _lerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Linear interpolation between two tensors.

        Uses PyTorch's optimized implementation for better performance.

        Args:
            a: Start tensor
            b: End tensor
            t: Interpolation factor in range [0, 1]

        Returns:
            Interpolated tensor: a + t * (b - a)
        """
        return torch.lerp(a, b, t)


@define
class PerlinNoise1D(_PerlinBase):
    """Single-octave 1D Perlin noise - optimized implementation."""

    def __call__(self) -> torch.Tensor:
        logging.debug(
            f"Generating 1D Perlin noise: scale={self.scale}, shape={self.shape}"
        )

        device = get_device()

        # Linear coordinates
        x_lin = torch.linspace(0.0, self.scale, self.shape[0], device=device)
        x0 = x_lin.to(torch.int32)
        xf = x_lin - x0

        # Gradient angles (using permutation table for determinism)
        grid_size = int(self.scale) + 2
        angles = torch.empty(grid_size, device=device).uniform_(0, 2 * torch.pi)

        # Compute noise at corners and interpolate
        n0 = torch.cos(angles[x0]) * xf
        n1 = torch.cos(angles[x0 + 1]) * (xf - 1)

        return self._lerp(n0, n1, self._fade(xf))


class PerlinNoise2D(_PerlinBase):
    """Single-octave 2D Perlin noise - optimized implementation."""

    def __call__(self) -> torch.Tensor:
        logging.debug(
            f"Generating 2D Perlin noise: scale={self.scale}, shape={self.shape}"
        )

        device = get_device()

        # Coordinate grid (more memory efficient for large grids)
        x_lin = torch.linspace(0.0, self.scale, self.shape[1], device=device)
        y_lin = torch.linspace(0.0, self.scale, self.shape[0], device=device)
        y, x = torch.meshgrid(y_lin, x_lin, indexing="ij")

        # Grid indices and fractional parts
        x0, y0 = x.to(torch.int32), y.to(torch.int32)
        x1, y1 = x0 + 1, y0 + 1
        xf, yf = x - x0, y - y0

        # Random gradient angles per grid cell
        grid_h, grid_w = int(self.scale) + 2, int(self.scale) + 2
        rotation = torch.empty((grid_h, grid_w), device=device).uniform_(
            0, 2 * torch.pi
        )

        @staticmethod
        def _dot(
            iy: torch.Tensor, ix: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor
        ) -> torch.Tensor:
            """Compute the directional derivative along an orientation field.

            This function calculates the dot product between the local gradient
            vector (dx, dy) and a unit direction vector defined by the rotation
            angle at each coordinate index. Effectively, it projects the gradient
            onto the specified orientation.

            Args:
                iy (torch.Tensor): Y-coordinates tensor for indexing into the rotation map.
                ix (torch.Tensor): X-coordinates tensor for indexing into the rotation map.
                dx (torch.Tensor): Gradient values in the x-direction.
                dy (torch.Tensor): Gradient values in the y-direction.

            Returns:
                torch.Tensor: A tensor containing the projected gradient magnitude
                              along the orientation angle `r` at each pixel location.
            """
            r = rotation[iy, ix]

            return torch.cos(r) * dx + torch.sin(r) * dy

        n00 = _dot(y0, x0, xf, yf)
        n10 = _dot(y0, x1, xf - 1, yf)
        n01 = _dot(y1, x0, xf, yf - 1)
        n11 = _dot(y1, x1, xf - 1, yf - 1)

        # Bilinear interpolation
        u = self._fade(xf)
        return self._lerp(
            self._lerp(n00, n10, u), self._lerp(n01, n11, u), self._fade(yf)
        )


class PerlinNoise3D(_PerlinBase):
    """Single-octave 3D Perlin noise - optimized implementation."""

    def __call__(self) -> torch.Tensor:
        logging.debug(
            f"Generating 3D Perlin noise: scale={self.scale}, shape={self.shape}"
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
        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
        xf, yf, zf = x - x0, y - y0, z - z0

        # 3D gradient vectors (random unit sphere directions)
        g = int(self.scale) + 2
        theta = torch.empty((g, g, g), device=device).uniform_(0, torch.pi)
        phi = torch.empty((g, g, g), device=device).uniform_(0, 2 * torch.pi)
        gx_vec = theta.sin() * phi.cos()
        gy_vec = theta.sin() * phi.sin()
        gz_vec = theta.cos()

        @staticmethod
        def _dot3(
            iz: torch.Tensor,
            iy: torch.Tensor,
            ix: torch.Tensor,
            dx: torch.Tensor,
            dy: torch.Tensor,
            dz: torch.Tensor,
        ) -> torch.Tensor:
            """Computes the dot product between the gradient field at specified voxel indices and a direction vector.

            Args:
                iz (torch.Tensor): Z-axis index tensor specifying the grid location.
                iy (torch.Tensor): Y-axis index tensor specifying the grid location.
                ix (torch.Tensor): X-axis index tensor specifying the grid location.
                dx (torch.Tensor): X-component of the direction vector.
                dy (torch.Tensor): Y-component of the direction vector.
                dz (torch.Tensor): Z-component of the direction vector.

            Returns:
                torch.Tensor: The computed dot product value(s) between the gradient and direction vectors.
            """
            return (
                gx_vec[iz, iy, ix] * dx
                + gy_vec[iz, iy, ix] * dy
                + gz_vec[iz, iy, ix] * dz
            )

        # 8 corner dot products
        n000 = _dot3(z0, y0, x0, xf, yf, zf)
        n100 = _dot3(z0, y0, x1, xf - 1, yf, zf)
        n010 = _dot3(z0, y1, x0, xf, yf - 1, zf)
        n110 = _dot3(z0, y1, x1, xf - 1, yf - 1, zf)
        n001 = _dot3(z1, y0, x0, xf, yf, zf - 1)
        n101 = _dot3(z1, y0, x1, xf - 1, yf, zf - 1)
        n011 = _dot3(z1, y1, x0, xf, yf - 1, zf - 1)
        n111 = _dot3(z1, y1, x1, xf - 1, yf - 1, zf - 1)

        # Trilinear interpolation
        u, v, w = self._fade(xf), self._fade(yf), self._fade(zf)
        x_interp0 = self._lerp(self._lerp(n000, n100, u), self._lerp(n010, n110, u), v)
        x_interp1 = self._lerp(self._lerp(n001, n101, u), self._lerp(n011, n111, u), v)
        return self._lerp(x_interp0, x_interp1, w)
