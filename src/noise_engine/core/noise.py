import logging
from typing import Optional, Tuple

import torch
from attrs import define, field, validators

from noise_engine.core.device import get_device


# ============================================================================
# PERLIN NOISE - SINGLE OCTAVE
# ============================================================================


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
class Perlin1D(_PerlinBase):
    """Single-octave 1D Perlin noise - optimized implementation."""

    def __call__(self) -> torch.Tensor:
        logging.debug(
            f"Generating 1D Perlin noise: scale={self.scale}, shape={self.shape}"
        )

        device = get_device()

        # Linear coordinates
        x_lin = torch.linspace(0.0, self.scale, self.shape[0], device=device)
        x0 = x_lin.long()
        xf = x_lin - x0

        # Gradient angles (using permutation table for determinism)
        grid_size = int(self.scale) + 2
        angles = torch.empty(grid_size, device=device).uniform_(0, 2 * torch.pi)

        # Compute noise at corners and interpolate
        n0 = angles[x0].cos() * xf
        n1 = angles[x0 + 1].cos() * (xf - 1)

        return self._lerp(n0, n1, self._fade(xf))


class Perlin2D(_PerlinBase):
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

        # Gradient dot products at four corners (vectorized)
        def dot(
            iy: torch.Tensor, ix: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor
        ) -> torch.Tensor:
            r = rotation[iy, ix]
            return r.cos() * dx + r.sin() * dy

        n00 = dot(y0, x0, xf, yf)
        n10 = dot(y0, x1, xf - 1, yf)
        n01 = dot(y1, x0, xf, yf - 1)
        n11 = dot(y1, x1, xf - 1, yf - 1)

        # Bilinear interpolation
        u = self._fade(xf)
        return self._lerp(
            self._lerp(n00, n10, u), self._lerp(n01, n11, u), self._fade(yf)
        )


class Perlin3D(_PerlinBase):
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

        # Dot product function for 3D gradients
        def dot3(iz, iy, ix, dx, dy, dz):
            return (
                gx_vec[iz, iy, ix] * dx
                + gy_vec[iz, iy, ix] * dy
                + gz_vec[iz, iy, ix] * dz
            )

        # 8 corner dot products
        n000 = dot3(z0, y0, x0, xf, yf, zf)
        n100 = dot3(z0, y0, x1, xf - 1, yf, zf)
        n010 = dot3(z0, y1, x0, xf, yf - 1, zf)
        n110 = dot3(z0, y1, x1, xf - 1, yf - 1, zf)
        n001 = dot3(z1, y0, x0, xf, yf, zf - 1)
        n101 = dot3(z1, y0, x1, xf - 1, yf, zf - 1)
        n011 = dot3(z1, y1, x0, xf, yf - 1, zf - 1)
        n111 = dot3(z1, y1, x1, xf - 1, yf - 1, zf - 1)

        # Trilinear interpolation
        u, v, w = self._fade(xf), self._fade(yf), self._fade(zf)
        x_interp0 = self._lerp(self._lerp(n000, n100, u), self._lerp(n010, n110, u), v)
        x_interp1 = self._lerp(self._lerp(n001, n101, u), self._lerp(n011, n111, u), v)
        return self._lerp(x_interp0, x_interp1, w)


# ============================================================================
# FRACTAL BROWNIAN MOTION (fBm) NOISE - MULTI-OCTAVE
# ============================================================================


@define
class _FractalNoiseBase:
    octaves: int
    scale: float
    shape: Tuple[int, ...]
    seed: Optional[int] = None
    turbulence: bool = False
    persistence: float = 0.5
    lacunarity: float = 2.0
    turbulence_gamma: float = 0.5

    def __attrs_post_init__(self):
        if len(self.shape) not in (1, 2, 3):
            raise ValueError(f"Shape must be 1D, 2D, or 3D. Got {len(self.shape)}D")
        if self.scale < 0:
            raise ValueError("Scale must be positive.")
        if self.seed is not None:
            torch.manual_seed(self.seed)
        if self.octaves < 0:
            raise ValueError("Octave must be positive.")


@define
class FractalNoise1D(_FractalNoiseBase):
    """Multi-octave 1D Perlin (fBm) noise generator."""

    def __call__(self) -> torch.Tensor:
        logging.debug(
            f"Generating 1D Fractal Noise: octaves={self.octaves}, shape={self.shape}"
        )

        device = get_device()
        total = torch.zeros(self.shape, device=device)
        current_scale = self.scale
        current_amp = 1.0

        for octave in range(self.octaves):
            layer_seed = self.seed + octave if self.seed is not None else None
            layer = Perlin1D(scale=current_scale, shape=self.shape, seed=layer_seed)()

            if self.turbulence:
                layer = torch.abs(layer) ** self.turbulence_gamma

            total += layer * current_amp
            current_amp *= self.persistence
            current_scale *= self.lacunarity

        return total


@define
class FractalNoise2D(_FractalNoiseBase):
    """Multi-octave 2D Perlin (fBm) noise generator."""

    def __call__(self) -> torch.Tensor:
        logging.debug(
            f"Generating 2D Fractal Noise: octaves={self.octaves}, shape={self.shape}"
        )

        device = get_device()
        total = torch.zeros(self.shape, device=device)
        current_scale = self.scale
        current_amp = 1.0

        for octave in range(self.octaves):
            layer_seed = self.seed + octave if self.seed is not None else None
            layer = Perlin2D(scale=current_scale, shape=self.shape, seed=layer_seed)()

            if self.turbulence:
                layer = torch.abs(layer) ** self.turbulence_gamma

            total += layer * current_amp
            current_amp *= self.persistence
            current_scale *= self.lacunarity

        return total


@define
class FractalNoise3D(_FractalNoiseBase):
    """Multi-octave 3D Perlin (fBm) noise generator."""

    def __call__(self) -> torch.Tensor:
        logging.debug(
            f"Generating 3D Fractal Noise: octaves={self.octaves}, shape={self.shape}"
        )

        device = get_device()
        total = torch.zeros(self.shape, device=device)
        current_scale = self.scale
        current_amp = 1.0

        for octave in range(self.octaves):
            layer_seed = self.seed + octave if self.seed is not None else None
            layer = Perlin3D(scale=current_scale, shape=self.shape, seed=layer_seed)()

            if self.turbulence:
                layer = torch.abs(layer) ** self.turbulence_gamma

            total += layer * current_amp
            current_amp *= self.persistence
            current_scale *= self.lacunarity

        return total


# ============================================================================
# WHITE NOISE - SIMPLE RANDOM GENERATORS
# ============================================================================


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


"""
TODO:
>>> Ridged/Billow Noise
>>> Curl Noise
>>> Worley Noise
>>> 3D Terrain Generator
"""

# fmt: off
# Permutation table (Ken Perlin's original 256-entry table)
_PERM_TABLE = [
    151,160,137, 91, 90, 15,131, 13,201, 95, 96, 53,194,233,  7,225,
    140, 36,103, 30, 69,142,  8, 99, 37,240, 21, 10, 23,190,  6,148,
    247,120,234, 75,  0, 26,197, 62, 94,252,219,203,117, 35, 11, 32,
     57,177, 33, 88,237,149, 56, 87,174, 20,125,136,171,168, 68,175,
     74,165, 71,134,139, 48, 27,166, 77,146,158,231, 83,111,229,122,
     60,211,133,230,220,105, 92, 41, 55, 46,245, 40,244,102,143, 54,
     65, 25, 63,161,  1,216, 80, 73,209, 76,132,187,208, 89, 18,169,
    200,196,135,130,116,188,159, 86,164,100,109,198,173,186,  3, 64,
     52,217,226,250,124,123,  5,202, 38,147,118,126,255, 82, 85,212,
    207,206, 59,227, 47, 16, 58, 17,182,189, 28, 42,223,183,170,213,
    119,248,152,  2, 44,154,163, 70,221,153,101,155,167, 43,172,  9,
    129, 22, 39,253, 19, 98,108,110, 79,113,224,232,178,185,112,104,
    218,246, 97,228,251, 34,242,193,238,210,144, 12,191,179,162,241,
     81, 51,145,235,249, 14,239,107, 49,192,214, 31,181,199,106,157,
    184, 84,204,176,115,121, 50, 45,127,  4,150,254,138,236,205, 93,
    222,114, 67, 29, 24, 72,243,141,128,195, 78, 66,215, 61,156,180,
]
# fmt: on


def _build_perm(seed: int | None, device: torch.device) -> torch.Tensor:
    """Build a 512-entry permutation table, optionally shuffled by seed."""
    perm = torch.tensor(_PERM_TABLE, dtype=torch.long, device=device)
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        idx = torch.randperm(256, generator=g, device=device)
        perm = perm[idx]
    # Double the table so we never need modulo in hot path
    return torch.cat([perm, perm])  # shape (512,)


# ---------------------------------------------------------------------------
# Gradient tables (pre-computed, stored as tensors)
# ---------------------------------------------------------------------------

# 1D: gradients are just ±1
_GRAD1 = torch.tensor([1.0, -1.0])

# 2D: 8 gradient directions (corners + edges of unit square, normalised)
_GRAD2 = torch.tensor(
    [
        [1.0, 1.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [-1.0, -1.0],
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
    ],
    dtype=torch.float32,
)

# 3D: 12 gradient directions (mid-points of a cube's edges)
_GRAD3 = torch.tensor(
    [
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
        [-1.0, -1.0, 0.0],
        [1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 0.0, -1.0],
        [0.0, 1.0, 1.0],
        [0.0, -1.0, 1.0],
        [0.0, 1.0, -1.0],
        [0.0, -1.0, -1.0],
    ],
    dtype=torch.float32,
)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


@define
class _SimplexBase:
    """Shared config for all simplex noise generators."""

    scale: float = field(default=1.0, validator=validators.gt(0))
    seed: int | None = field(default=None)

    def _perm(self, device: torch.device) -> torch.Tensor:
        return _build_perm(self.seed, device)

    def _grad1(self, device: torch.device) -> torch.Tensor:
        return _GRAD1.to(device)

    def _grad2(self, device: torch.device) -> torch.Tensor:
        return _GRAD2.to(device)

    def _grad3(self, device: torch.device) -> torch.Tensor:
        return _GRAD3.to(device)


# ---------------------------------------------------------------------------
# Simplex1D
# ---------------------------------------------------------------------------


@define
class Simplex1D(_SimplexBase):
    """
    1D Simplex noise.

    Returns a tensor of shape `(n,)` with values in approximately [-1, 1].

    Parameters
    ----------
    n     : number of output samples
    scale : wavelength in samples (larger → smoother)
    seed  : optional RNG seed for reproducibility
    """

    n: int = field(default=256, validator=validators.gt(0))

    def __call__(self) -> torch.Tensor:
        logging.debug(f"Simplex1D: n={self.n}, scale={self.scale}")
        device = get_device()
        perm = self._perm(device)
        grad1 = self._grad1(device)

        # Sample coordinates
        x = torch.arange(self.n, dtype=torch.float32, device=device) / self.scale

        # Integer cell and fractional offset
        i0 = x.floor().long()
        x0 = x - i0.float()
        x1 = x0 - 1.0

        # Permuted gradient indices
        gi0 = perm[i0 & 255] & 1  # mod 2 → index into grad1
        gi1 = perm[(i0 + 1) & 255] & 1

        # Gradient lookup
        g0 = grad1[gi0]  # (n,)
        g1 = grad1[gi1]  # (n,)

        # Kernel: t = 1 - x², clamped to [0, 1], then t⁴
        t0 = (1.0 - x0 * x0).clamp(min=0.0) ** 4
        t1 = (1.0 - x1 * x1).clamp(min=0.0) ** 4

        # Contribution from each corner
        n0 = t0 * (g0 * x0)
        n1 = t1 * (g1 * x1)

        # Scale to approx [-1, 1]
        return (n0 + n1) * 0.395 * 2.0


# ---------------------------------------------------------------------------
# Simplex2D
# ---------------------------------------------------------------------------

# Skewing constants for 2D
_F2 = 0.5 * (3.0**0.5 - 1.0)  # skew
_G2 = (3.0 - 3.0**0.5) / 6.0  # unskew


@define
class Simplex2D(_SimplexBase):
    """
    2D Simplex noise.

    Returns a tensor of shape `(H, W)` with values in approximately [-1, 1].

    Parameters
    ----------
    shape : (H, W) output resolution
    scale : wavelength in pixels (larger → smoother)
    seed  : optional RNG seed for reproducibility
    """

    shape: tuple[int, int] = field(default=(256, 256))

    @shape.validator
    def _check_shape(self, attribute, value):
        if len(value) != 2 or any(v <= 0 for v in value):
            raise ValueError("shape must be (H, W) with positive integers")

    def __call__(self) -> torch.Tensor:
        H, W = self.shape
        logging.debug(f"Simplex2D: shape={self.shape}, scale={self.scale}")
        device = get_device()
        perm = self._perm(device)
        grad2 = self._grad2(device)

        # Build coordinate grid
        ys = torch.arange(H, dtype=torch.float32, device=device) / self.scale
        xs = torch.arange(W, dtype=torch.float32, device=device) / self.scale
        y, x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W) each

        # --------------- Skew input space to determine simplex cell --------
        s = (x + y) * _F2
        i = (x + s).floor().long()  # cell coords
        j = (y + s).floor().long()

        t = (i + j).float() * _G2
        # Unskew back — corner 0 in original space
        X0 = i.float() - t
        Y0 = j.float() - t
        x0 = x - X0
        y0 = y - Y0

        # Determine which simplex triangle we're in
        # i1, j1 = second corner offset (either (1,0) or (0,1))
        i1 = (x0 >= y0).long()
        j1 = 1 - i1

        # Offsets for corners 1 and 2
        x1 = x0 - i1.float() + _G2
        y1 = y0 - j1.float() + _G2
        x2 = x0 - 1.0 + 2.0 * _G2
        y2 = y0 - 1.0 + 2.0 * _G2

        # --------------- Gradient index lookup ----------------------------
        ii = i & 255
        jj = j & 255

        gi0 = perm[ii + perm[jj]] % 8
        gi1 = perm[ii + i1 + perm[jj + j1]] % 8
        gi2 = perm[ii + 1 + perm[jj + 1]] % 8

        g0 = grad2[gi0]  # (H, W, 2)
        g1 = grad2[gi1]
        g2 = grad2[gi2]

        # --------------- Kernel & contributions ---------------------------
        n0 = (0.5 - x0 * x0 - y0 * y0).clamp(min=0.0) ** 4 * (
            g0[..., 0] * x0 + g0[..., 1] * y0
        )
        n1 = (0.5 - x1 * x1 - y1 * y1).clamp(min=0.0) ** 4 * (
            g1[..., 0] * x1 + g1[..., 1] * y1
        )
        n2 = (0.5 - x2 * x2 - y2 * y2).clamp(min=0.0) ** 4 * (
            g2[..., 0] * x2 + g2[..., 1] * y2
        )

        # Scale to approx [-1, 1]
        return (n0 + n1 + n2) * 70.0


# ---------------------------------------------------------------------------
# Simplex3D
# ---------------------------------------------------------------------------

# Skewing constants for 3D
_F3 = 1.0 / 3.0
_G3 = 1.0 / 6.0


@define
class Simplex3D(_SimplexBase):
    """
    3D Simplex noise.

    Returns a tensor of shape `(D, H, W)` with values in approximately [-1, 1].

    Parameters
    ----------
    shape : (D, H, W) output resolution
    scale : wavelength in voxels (larger → smoother)
    seed  : optional RNG seed for reproducibility
    """

    shape: tuple[int, int, int] = field(default=(64, 64, 64))

    @shape.validator
    def _check_shape(self, attribute, value):
        if len(value) != 3 or any(v <= 0 for v in value):
            raise ValueError("shape must be (D, H, W) with positive integers")

    def __call__(self) -> torch.Tensor:
        D, H, W = self.shape
        logging.debug(f"Simplex3D: shape={self.shape}, scale={self.scale}")
        device = get_device()
        perm = self._perm(device)
        grad3 = self._grad3(device)  # (12, 3)

        # Coordinate grid
        zs = torch.arange(D, dtype=torch.float32, device=device) / self.scale
        ys = torch.arange(H, dtype=torch.float32, device=device) / self.scale
        xs = torch.arange(W, dtype=torch.float32, device=device) / self.scale
        z, y, x = torch.meshgrid(zs, ys, xs, indexing="ij")  # (D, H, W) each

        # --------------- Skew -----------------------------------------------
        s = (x + y + z) * _F3
        i = (x + s).floor().long()
        j = (y + s).floor().long()
        k = (z + s).floor().long()

        t = (i + j + k).float() * _G3
        X0 = i.float() - t
        Y0 = j.float() - t
        Z0 = k.float() - t
        x0 = x - X0
        y0 = y - Y0
        z0 = z - Z0

        # --------------- Simplex ordering -----------------------------------
        # Determine which of the 6 tetrahedral simplices we're in
        x_ge_y = x0 >= y0
        y_ge_z = y0 >= z0
        x_ge_z = x0 >= z0

        # Corner 1
        i1 = (x_ge_y & x_ge_z).long()
        j1 = (~x_ge_y & y_ge_z).long()
        k1 = (~x_ge_z & ~y_ge_z).long()

        # Corner 2
        i2 = (x_ge_y | x_ge_z).long()
        j2 = (~x_ge_y | y_ge_z).long()
        k2 = (~(x_ge_z & y_ge_z)).long()

        # Offsets for corners 1, 2, 3
        x1 = x0 - i1.float() + _G3
        y1 = y0 - j1.float() + _G3
        z1 = z0 - k1.float() + _G3

        x2 = x0 - i2.float() + 2.0 * _G3
        y2 = y0 - j2.float() + 2.0 * _G3
        z2 = z0 - k2.float() + 2.0 * _G3

        x3 = x0 - 1.0 + 3.0 * _G3
        y3 = y0 - 1.0 + 3.0 * _G3
        z3 = z0 - 1.0 + 3.0 * _G3

        # --------------- Gradient indices -----------------------------------
        ii = i & 255
        jj = j & 255
        kk = k & 255

        gi0 = perm[ii + perm[jj + perm[kk]]] % 12
        gi1 = perm[ii + i1 + perm[jj + j1 + perm[kk + k1]]] % 12
        gi2 = perm[ii + i2 + perm[jj + j2 + perm[kk + k2]]] % 12
        gi3 = perm[ii + 1 + perm[jj + 1 + perm[kk + 1]]] % 12

        g0 = grad3[gi0]  # (D, H, W, 3)
        g1 = grad3[gi1]
        g2 = grad3[gi2]
        g3 = grad3[gi3]

        # --------------- Contributions --------------------------------------
        def corner(g, dx, dy, dz):
            t_ = (0.6 - dx * dx - dy * dy - dz * dz).clamp(min=0.0) ** 4
            dot = g[..., 0] * dx + g[..., 1] * dy + g[..., 2] * dz
            return t_ * dot

        n0 = corner(g0, x0, y0, z0)
        n1 = corner(g1, x1, y1, z1)
        n2 = corner(g2, x2, y2, z2)
        n3 = corner(g3, x3, y3, z3)

        # Scale to approx [-1, 1]
        return (n0 + n1 + n2 + n3) * 32.0


# ============================================================================
# FACTORY FUNCTION - EASY USAGE
# ============================================================================


def noise_factory(
    noise_type: str = "perlin",
    dimensions: int = 2,
    scale: float = 10.0,
    shape=None,
    octaves: int = 1,
    seed: Optional[int] = None,
    **kwargs,
) -> object:
    """
    Factory function for creating noise generators.

    Args:
        noise_type: Type of noise ("perlin", "simplex", "white")
        dimensions: Number of dimensions (1, 2, or 3)
        scale: Spatial frequency
        shape: Output shape tuple
        octaves: Number of octaves for fractal noise (>1 enables fBm)
        seed: Random seed
        **kwargs: Additional parameters (persistence, lacunarity, etc.)

    Returns:
        Configured noise generator instance

    Example:
        >>> gen = noise_factory("simplex", dimensions=2, scale=10.0, shape=(512, 512))
        >>> noise = gen()

        >>> gen = noise_factory("perlin", dimensions=3, scale=20.0,
        ...                     shape=(128, 128, 128), octaves=4)
        >>> noise = gen()
    """
    if shape is None:
        raise ValueError("Shape must be specified")

    type_lower = noise_type.lower()

    # Single octave noise
    _SINGLE_OCTAVE = {
        ("perlin", 1): lambda scale, shape, seed, **_: Perlin1D(scale, shape, seed),
        ("perlin", 2): lambda scale, shape, seed, **_: Perlin2D(scale, shape, seed),
        ("perlin", 3): lambda scale, shape, seed, **_: Perlin3D(scale, shape, seed),
        ("simplex", 2): lambda scale, shape, seed, **_: Simplex2D(scale, shape, seed),
        ("simplex", 3): lambda scale, shape, seed, **_: Simplex3D(scale, shape, seed),
        ("white", 1): lambda shape, seed, **_: WhiteNoise1D(shape, seed),
        ("white", 2): lambda shape, seed, **_: WhiteNoise2D(shape, seed),
        ("white", 3): lambda shape, seed, **_: WhiteNoise3D(shape, seed),
    }

    # Fractal noise (fBm)
    _FRACTAL = {
        ("perlin", 1): lambda scale, octaves, shape, seed, **kw: FractalNoise1D(
            scale, octaves, shape, seed=seed, **kw
        ),
        ("perlin", 2): lambda scale, octaves, shape, seed, **kw: FractalNoise2D(
            scale, octaves, shape, seed=seed, **kw
        ),
        ("perlin", 3): lambda scale, octaves, shape, seed, **kw: FractalNoise3D(
            scale, octaves, shape, seed=seed, **kw
        ),
    }

    type_lower = noise_type.lower()
    key = (type_lower, dimensions)

    if octaves == 1:
        factory = _SINGLE_OCTAVE.get(key)
        if factory:
            return factory(scale, shape, seed)
    else:
        factory = _FRACTAL.get(key)
        if factory:
            return factory(scale, octaves, shape, seed, **kwargs)

    raise ValueError(
        f"Unsupported noise_type='{noise_type}' with dimensions={dimensions}"
    )
