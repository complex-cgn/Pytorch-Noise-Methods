import logging
from dataclasses import dataclass, field
from typing import ClassVar, Optional
from noise_engine.utils.noise_utils import fade, lerp

import torch
from noise_engine.core.device import device


@dataclass
class Perlin1D:
    """Single-octave 1D Perlin noise."""

    scale: float
    shape: tuple[int]
    seed: Optional[int] = None

    def __call__(self) -> torch.Tensor:
        """Compute one octave of 1D Perlin noise.

        Returns:
            Noise tensor of shape (length,) in approximately [-1, 1].
        """
        logging.debug(
            f"Generating 1D Perlin noise with scale={self.scale}, shape={self.shape}, seed={self.seed}"
        )

        x_lin = torch.linspace(0.0, self.scale, self.shape[0], device=device)
        x0 = x_lin.long()
        xf = x_lin - x0

        grid_w = int(self.scale) + 2
        angles = torch.empty(grid_w, device=device).uniform_(0, 2 * torch.pi)

        n0 = angles[x0].cos() * xf
        n1 = angles[x0 + 1].cos() * (xf - 1)
        return lerp(n0, n1, fade(xf))


@dataclass
class Perlin2D:
    """Single-octave 2D Perlin noise generator."""

    scale: float
    shape: tuple[int, int]
    seed: Optional[int] = None

    def __post_init__(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)

    def __call__(self) -> torch.Tensor:
        """Compute one octave of Perlin noise.

        Returns:
            Noise tensor of shape (height, width) in approximately [-1, 1].
        """

        logging.debug(
            f"Generating 2D Perlin noise with scale={self.scale}, shape={self.shape}, seed={self.seed}"
        )

        # coordinate grid
        x_lin = torch.linspace(0.0, self.scale, self.shape[1], device=device)
        y_lin = torch.linspace(0.0, self.scale, self.shape[0], device=device)
        y, x = torch.meshgrid(y_lin, x_lin, indexing="ij")

        # random gradient angles per grid cell
        grid_h = int(self.scale) + 2
        grid_w = int(self.scale) + 2
        rotation = torch.empty((grid_h, grid_w), device=device).uniform_(
            0, 2 * torch.pi
        )

        # grid indices and fractional offsets
        x0, y0 = x.long(), y.long()
        x1, y1 = x0 + 1, y0 + 1
        xf, yf = x - x0, y - y0

        # gradient dot products at four corners
        def dot(iy, ix, dx, dy):
            r = rotation[iy, ix]
            return r.cos() * dx + r.sin() * dy

        n00 = dot(y0, x0, xf, yf)
        n10 = dot(y0, x1, xf - 1, yf)
        n01 = dot(y1, x0, xf, yf - 1)
        n11 = dot(y1, x1, xf - 1, yf - 1)

        # interpolate
        u = fade(xf)
        return lerp(lerp(n00, n10, u), lerp(n01, n11, u), fade(yf))


@dataclass
class Perlin3D:
    """Single-octave 3D Perlin noise."""

    scale: float
    shape: tuple[int, int, int]
    seed: Optional[int] = None

    def __call__(self) -> torch.Tensor:
        """Compute one octave of 3D Perlin noise.

        Returns:
            Noise tensor of shape (depth, height, width) in approximately [-1, 1].
        """

        logging.debug(
            f"Generating 3D Perlin noise with scale={self.scale}, shape={self.shape}, seed={self.seed}"
        )

        # z, y, x grid
        sz, sy, sx = self.shape
        z_lin = torch.linspace(0.0, self.scale, sz, device=device)
        y_lin = torch.linspace(0.0, self.scale, sy, device=device)
        x_lin = torch.linspace(0.0, self.scale, sx, device=device)
        z, y, x = torch.meshgrid(z_lin, y_lin, x_lin, indexing="ij")

        g = int(self.scale) + 2
        # 3D gradient vectors — random vectors on the unit sphere instead of angles
        theta = torch.empty((g, g, g), device=device).uniform_(0, torch.pi)
        phi = torch.empty((g, g, g), device=device).uniform_(0, 2 * torch.pi)
        gx = theta.sin() * phi.cos()
        gy = theta.sin() * phi.sin()
        gz = theta.cos()

        x0, y0, z0 = x.long(), y.long(), z.long()
        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
        xf, yf, zf = x - x0, y - y0, z - z0

        def dot3(iz, iy, ix, dx, dy, dz):
            return gx[iz, iy, ix] * dx + gy[iz, iy, ix] * dy + gz[iz, iy, ix] * dz

        # 8 corner gradients and their dot products
        n000 = dot3(z0, y0, x0, xf, yf, zf)
        n100 = dot3(z0, y0, x1, xf - 1, yf, zf)
        n010 = dot3(z0, y1, x0, xf, yf - 1, zf)
        n110 = dot3(z0, y1, x1, xf - 1, yf - 1, zf)
        n001 = dot3(z1, y0, x0, xf, yf, zf - 1)
        n101 = dot3(z1, y0, x1, xf - 1, yf, zf - 1)
        n011 = dot3(z1, y1, x0, xf, yf - 1, zf - 1)
        n111 = dot3(z1, y1, x1, xf - 1, yf - 1, zf - 1)

        u, v, w = fade(xf), fade(yf), fade(zf)
        x_interp0 = lerp(lerp(n000, n100, u), lerp(n010, n110, u), v)
        x_interp1 = lerp(lerp(n001, n101, u), lerp(n011, n111, u), v)
        return lerp(x_interp0, x_interp1, w)


@dataclass
class FractalNoise1D:
    """Multi-octave 1D Perlin (fBm) noise generator."""

    scale: float
    octaves: int
    shape: tuple[int]
    turbulence: bool = False
    persistence: float = 0.5
    lacunarity: float = 2.0
    turbulence_gamma: float = 0.5
    seed: Optional[int] = None

    def __call__(self) -> torch.Tensor:
        """Generate 1D fractal Brownian motion noise.

        Returns:
            Combined noise tensor of shape (length,).
        """

        logging.debug(
            f"Generating 1D Fractal Noise with scale={self.scale}, octaves={self.octaves}, shape={self.shape}, turbulence={self.turbulence}, persistence={self.persistence}, lacunarity={self.lacunarity}, turbulence_gamma={self.turbulence_gamma}, seed={self.seed}"
        )

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


@dataclass
class FractalNoise2D:
    """Multi-octave 2D Perlin (fBm) noise generator."""

    scale: float
    octaves: int
    shape: tuple[int, int]
    turbulence: bool = False
    persistence: float = 0.5
    lacunarity: float = 2.0
    turbulence_gamma: float = 0.5
    seed: Optional[int] = None

    def __call__(self) -> torch.Tensor:
        """Generate fractal Brownian motion noise.

        Returns:
            Combined noise tensor of shape (height, width).
        """

        logging.debug(
            f"Generating 2D Fractal Noise with scale={self.scale}, octaves={self.octaves}, shape={self.shape}, turbulence={self.turbulence}, persistence={self.persistence}, lacunarity={self.lacunarity}, turbulence_gamma={self.turbulence_gamma}, seed={self.seed}"
        )

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


@dataclass
class FractalNoise3D:
    """Multi-octave 3D Perlin (fBm) noise generator."""

    scale: float
    octaves: int
    shape: tuple[int, int, int]
    turbulence: bool = False
    persistence: float = 0.5
    lacunarity: float = 2.0
    turbulence_gamma: float = 0.5
    seed: Optional[int] = None

    def __call__(self) -> torch.Tensor:
        """Generate 3D fractal Brownian motion noise.

        Returns:
            Combined noise tensor of shape (depth, height, width).
        """

        logging.debug(
            f"Generating 3D Fractal Noise with scale={self.scale}, octaves={self.octaves}, shape={self.shape}, turbulence={self.turbulence}, persistence={self.persistence}, lacunarity={self.lacunarity}, turbulence_gamma={self.turbulence_gamma}, seed={self.seed}"
        )

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


@dataclass
class WhiteNoise1D:
    """Simple white noise generator.

    Generates random values uniformly distributed in [0, 1] for each pixel.

    Example:
        noise = WhiteNoise1D(shape=(256,))()
    """

    shape: tuple[int]
    _buffer: ClassVar[torch.Tensor] = field(init=False, repr=False)

    def __post_init__(self):
        self._buffer = torch.empty(self.shape, device=device)

    def __call__(self) -> torch.Tensor:
        """Fill buffer with uniform random values and return it.

        Returns:
            Tensor of shape (H,) with values in [0, 1].
        """

        logging.debug(f"Generating 1D White Noise with shape={self.shape}")

        return self._buffer.uniform_()


@dataclass
class WhiteNoise2D:
    """Simple white noise generator.

    Generates random values uniformly distributed in [0, 1] for each pixel.

    Example:
        noise = WhiteNoise2D(shape=(256, 256))()
    """

    shape: tuple[int, int]
    _buffer: ClassVar[torch.Tensor] = field(init=False, repr=False)

    def __post_init__(self):
        self._buffer = torch.empty(self.shape, device=device)

    def __call__(self) -> torch.Tensor:
        """Fill buffer with uniform random values and return it.

        Returns:
            Tensor of shape (H, W) with values in [0, 1].
        """

        logging.debug(f"Generating 2D White Noise with shape={self.shape}")

        return self._buffer.uniform_()


class WhiteNoise3D:
    """Simple white noise generator.

    Generates random values uniformly distributed in [0, 1] for each pixel.

    Example:
        noise = WhiteNoise3D(shape=(256, 256, 256))()
    """

    shape: tuple[int, int, int]
    _buffer: ClassVar[torch.Tensor] = field(init=False, repr=False)

    def __post_init__(self):
        self._buffer = torch.empty(self.shape, device=device)

    def __call__(self) -> torch.Tensor:
        """Fill buffer with uniform random values and return it.

        Returns:
            Tensor of shape (D, H, W) with values in [0, 1].
        """

        logging.debug(f"Generating 3D White Noise with shape={self.shape}")

        return self._buffer.uniform_()
