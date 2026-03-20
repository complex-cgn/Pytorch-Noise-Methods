import logging
from typing import Optional, Tuple

import torch
from attrs import define, field

from noise_engine.core.device import get_device
from noise_engine.utils.noise_utils import fade, lerp


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

        return lerp(n0, n1, fade(xf))


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
        u = fade(xf)
        return lerp(lerp(n00, n10, u), lerp(n01, n11, u), fade(yf))


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
        u, v, w = fade(xf), fade(yf), fade(zf)
        x_interp0 = lerp(lerp(n000, n100, u), lerp(n010, n110, u), v)
        x_interp1 = lerp(lerp(n001, n101, u), lerp(n011, n111, u), v)
        return lerp(x_interp0, x_interp1, w)


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


@define
class Simplex2D:
    """
    2D Simplex noise generator.

    Simplex noise is ~40% faster than Perlin noise for 2D and produces
    fewer artifacts with more natural-looking patterns.

    Args:
        scale: Spatial frequency of the noise
        shape: Output shape (height, width)
        seed: Random seed for reproducibility

    Example:
        >>> simplex = Simplex2D(scale=10.0, shape=(256, 256), seed=42)
        >>> noise = simplex()  # Returns tensor of shape (256, 256)
    """

    scale: float
    shape: tuple[int, int]
    seed: Optional[int] = None

    def __call__(self) -> torch.Tensor:
        logging.debug(
            f"Generating 2D Simplex Noise: scale={self.scale}, shape={self.shape}"
        )

        device = get_device()
        height, width = self.shape

        # Skewing factors for 2D simplex grid
        F2 = 0.5 * (torch.sqrt(torch.tensor(3.0)) - 1.0)
        G2 = (3.0 - torch.sqrt(torch.tensor(3.0))) / 6.0

        # Create coordinate grid
        x_lin = torch.linspace(0.0, self.scale, width, device=device)
        y_lin = torch.linspace(0.0, self.scale, height, device=device)
        y_grid, x_grid = torch.meshgrid(y_lin, x_lin, indexing="ij")

        # Skew input space to determine simplex cell
        s = (x_grid + y_grid) * F2
        xi = x_grid.long() + s.to(torch.int32)
        yi = y_grid.long() + s.to(torch.int32)

        # Unskew back to Cartesian coordinates
        t = (xi + yi).to(torch.float32) * G2
        x0 = x_grid - t
        y0 = y_grid - t

        # Determine which simplex we're in
        i1 = (x0 > y0).to(torch.int32)
        j1 = 1 - i1

        # Offsets for corners of simplex
        x1 = x0 + i1.to(torch.float32) - G2
        y1 = y0 + j1.to(torch.float32) - G2
        x2 = x0 + 1.0 - 2.0 * G2
        y2 = y0 + 1.0 - 2.0 * G2

        # Create permutation table for gradients
        if self.seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
            perm = torch.randperm(256, generator=generator).to(torch.int32)
        else:
            perm = torch.randperm(256).to(torch.int32)

        # Gradient indices for each corner
        ii = xi % 256
        jj = yi % 256

        def grad(ix, iy, x_in, y_in):
            """Compute gradient contribution."""
            p = (perm[(ix + perm[iy]) % 256] % 12).to(torch.float32)
            # 12 possible gradients in 2D simplex noise
            h = torch.where(p < 8, 0.5, 0.0)
            x = torch.where((p & 4) > 0, 1.0, -1.0) * (x_in + h)
            y = torch.where((p & 2) > 0, 1.0, -1.0) * (y_in + h)

            # Select which component based on p
            x = torch.where(p < 4, x, y)
            y = torch.where(p < 4, y, x)
            return x * x_in + y * y_in

        # Calculate contributions from each corner
        n0 = grad(ii, jj, x0, y0)
        n1 = grad(ii + i1, jj + j1, x1, y1)
        n2 = grad(ii + 1, jj + 1, x2, y2)

        # Distance from corners (squared)
        t0 = 0.5 - x0 * x0 - y0 * y0
        t1 = 0.5 - x1 * x1 - y1 * y1
        t2 = 0.5 - x2 * x2 - y2 * y2

        # Smooth falloff function (quartic)
        t0 = torch.clamp(t0, 0.0).pow(4) * n0
        t1 = torch.clamp(t1, 0.0).pow(4) * n1
        t2 = torch.clamp(t2, 0.0).pow(4) * n2

        # Sum contributions and scale
        return 70.0 * (t0 + t1 + t2)


@define
class Simplex3D:
    """
    3D Simplex noise generator.

    Args:
        scale: Spatial frequency of the noise
        shape: Output shape (depth, height, width)
        seed: Random seed for reproducibility
    """

    scale: float
    shape: tuple[int, int, int]
    seed: Optional[int] = None

    def __call__(self) -> torch.Tensor:
        logging.debug(
            f"Generating 3D Simplex Noise: scale={self.scale}, shape={self.shape}"
        )

        device = get_device()
        depth, height, width = self.shape

        # Skewing factors for 3D simplex grid
        F3 = 1.0 / 3.0
        G3 = 1.0 / 6.0

        # Create coordinate grids
        x_lin = torch.linspace(0.0, self.scale, width, device=device)
        y_lin = torch.linspace(0.0, self.scale, height, device=device)
        z_lin = torch.linspace(0.0, self.scale, depth, device=device)
        z_grid, y_grid, x_grid = torch.meshgrid(z_lin, y_lin, x_lin, indexing="ij")

        # Skew input space to determine simplex cell
        s = (x_grid + y_grid + z_grid) * F3
        xi = x_grid.long() + s.to(torch.int32)
        yi = y_grid.long() + s.to(torch.int32)
        zi = z_grid.long() + s.to(torch.int32)

        # Unskew back to Cartesian coordinates
        t = (xi + yi + zi).to(torch.float32) * G3
        x0 = x_grid - t
        y0 = y_grid - t
        z0 = z_grid - t

        # Determine simplex cell ordering
        i1 = (x0 > y0).to(torch.int32)
        j1 = (y0 > z0).to(torch.int32)
        i2 = (i1 > j1).to(torch.int32)

        # Offsets for remaining corners
        x1 = x0 + i2 - G3
        y1 = y0 + i2 - 1.0 + G3
        z1 = z0 + i2 - 1.0 + G3
        x2 = x0 + j1 * (1.0 - i2) - 2.0 * G3
        y2 = y0 + j1 * (1.0 - i2) - 1.0 + G3
        z2 = z0 + j1 * (i2 - 1.0) - 1.0 + G3
        x3 = x0 + 1.0 - 3.0 * G3
        y3 = y0 + 1.0 - 3.0 * G3
        z3 = z0 + 1.0 - 3.0 * G3

        # Permutation table
        if self.seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
            perm = torch.randperm(256, generator=generator).to(torch.int32)
        else:
            perm = torch.randperm(256).to(torch.int32)

        # Gradient indices
        ii = xi % 256
        jj = yi % 256
        kk = zi % 256

        # Klasik 12 gradient: küpün kenar orta noktaları
        GRAD3 = torch.tensor(
            [
                (1, 1, 0),
                (-1, 1, 0),
                (1, -1, 0),
                (-1, -1, 0),
                (1, 0, 1),
                (-1, 0, 1),
                (1, 0, -1),
                (-1, 0, -1),
                (0, 1, 1),
                (0, -1, 1),
                (0, 1, -1),
                (0, -1, -1),
            ],
            dtype=torch.float32,
        )

        def grad(ix, iy, iz, x_in, y_in, z_in):
            # Permutation lookup — hangi gradient index'i seçilecek
            p = perm[(ix + perm[(iy + perm[iz % 256]) % 256]) % 256] % 12
            # p shape: (...) — her nokta için bir gradient index

            # Gradient vektörlerini lookup et
            g = GRAD3[p]  # shape: (..., 3)

            # Dot product: g · (x_in, y_in, z_in)
            return g[..., 0] * x_in + g[..., 1] * y_in + g[..., 2] * z_in

        # Calculate contributions from all 8 corners of simplex
        n0 = grad(ii, jj, kk, x0, y0, z0)
        n1 = grad(ii + i2, jj + i2 - 1, kk + i2 - 1, x1, y1, z1)
        n2 = grad(
            ii + j1 * (i2 - 1) + 1,
            jj + j1 * (i2 - 1),
            kk + j1 * (i2 - 1) - 1,
            x2,
            y2,
            z2,
        )
        n3 = grad(ii + i2 - 1 + 1, jj + i2 - 1, kk + i2 - 1, x3, y3, z3)

        # Distance falloff (quartic)
        t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0
        t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1
        t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2
        t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3

        t0 = torch.clamp(t0, 0.0).pow(4) * n0
        t1 = torch.clamp(t1, 0.0).pow(4) * n1
        t2 = torch.clamp(t2, 0.0).pow(4) * n2
        t3 = torch.clamp(t3, 0.0).pow(4) * n3

        return 32.0 * (t0 + t1 + t2 + t3)


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
