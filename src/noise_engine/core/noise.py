import logging
from typing import Optional

import torch
from dataclasses import dataclass, field

from noise_engine.core.device import device
from noise_engine.settings import settings


def fade(t: torch.Tensor) -> torch.Tensor:
    """
    Compute the fade curve for smooth interpolation.

    Uses the quintic polynomial: t³ * (t * (6*t - 15) + 10)

    Args:
        t: Input tensor in range [0, 1]
    Returns:
        Smoothed tensor in range [0, 1]
    """
    return t * t * t * (t * (t * 6 - 15) + 10)


def lerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Linear interpolation between two tensors.

    Args:
        a: First tensor
        b: Second tensor
        t: Interpolation factor in range [0, 1]

    Returns:
        Interpolated tensor
    """
    return a + t * (b - a)


@dataclass
class PerlinState:
    shape: tuple[int, int]
    device: torch.device = device
    grids: dict[str, torch.Tensor] = field(init=False, repr=False)

    def __post_init__(self):
        self.grids = {
            "n00": torch.empty(self.shape, device=self.device),
            "n10": torch.empty(self.shape, device=self.device),
            "n01": torch.empty(self.shape, device=self.device),
            "n11": torch.empty(self.shape, device=self.device),
        }


@dataclass
class Perlin2D:
    scale: float
    seed: Optional[int] = None
    shape: tuple[int, int] = (settings.noise.width, settings.noise.height)

    # Precompute noise state to avoid redundant allocations during multi-octave generation
    state: PerlinState = field(init=False, repr=False)

    def __post_init__(self):
        self.state = PerlinState(self.shape, device=device)

    def __call__(
        self,
    ):
        """
        Compute Perlin noise for a single octave.

        Args:
            seed: Random seed for this octave
            scale: Spatial scale

        Returns:
            Noise tensor of shape (height, width)
        """

        n00 = self.state.grids["n00"]
        n10 = self.state.grids["n10"]
        n01 = self.state.grids["n01"]
        n11 = self.state.grids["n11"]

        # Create coordinate grid
        logging.debug(
            f"Computing noise grid with scale {self.scale} and seed {self.seed}"
        )
        x_lin = torch.linspace(0.0, self.scale, settings.noise.width, device=device)
        y_lin = torch.linspace(0.0, self.scale, settings.noise.height, device=device)
        y, x = torch.meshgrid(y_lin, x_lin, indexing="ij")

        # Generate random rotation matrix
        logging.debug("Generating random rotation matrix for noise gradients")
        grid_w = int(self.scale) + 2
        grid_h = int(self.scale) + 2
        rotation = torch.empty((grid_h, grid_w), device=device).uniform_(
            0, 2 * torch.pi
        )

        # Convert coordinates to grid indices
        logging.debug("Calculating grid indices and fractional offsets")
        x0 = x.to(torch.int64)
        y0 = y.to(torch.int64)
        x1 = x0 + 1
        y1 = y0 + 1

        # Compute fractional parts
        xf = x - x0
        yf = y - y0

        # Compute dot products for each corner
        logging.debug("Computing gradient contributions for simplex corners")
        r00 = rotation[y0, x0]
        r10 = rotation[y0, x1]
        r01 = rotation[y1, x0]
        r11 = rotation[y1, x1]

        c00, s00 = r00.cos(), r00.sin()
        c10, s10 = r10.cos(), r10.sin()
        c01, s01 = r01.cos(), r01.sin()
        c11, s11 = r11.cos(), r11.sin()

        xf0 = xf - 1
        yf0 = yf - 1

        n00 = c00 * xf + s00 * yf
        n10 = c10 * xf0 + s10 * yf
        n01 = c01 * xf + s01 * yf0
        n11 = c11 * xf0 + s11 * yf0

        # Interpolate
        logging.debug("Performing fade and linear interpolation")
        u = fade(xf)
        value = lerp(
            lerp(n00, n10, u),
            lerp(n01, n11, u),
            fade(yf),
        )

        return value


@dataclass
class FractalNoise2D:
    scale: float
    octaves: int
    persistence: float
    lacunarity: float
    turbulence: bool
    seed: Optional[int] = None
    shape: tuple[int, int] = (settings.noise.width, settings.noise.height)

    def __call__(self) -> torch.Tensor:
        """
        Generate multi-octave Perlin noise.

        Args:
            octaves: Number of noise layers to sum
            persistence: Amplitude decay per octave (typically 0.5)
            lacunarity: Frequency multiplier per octave (typically 2.0)
            turbulence: Whether to use turbulence mode

        Returns:
            Combined noise tensor of shape (height, width)
        """
        total_noise = torch.zeros(
            (settings.noise.height, settings.noise.width), device=device
        )
        current_scale = settings.noise.scale
        current_amp = 0.1

        for octave in range(self.octaves):
            logging.debug(f"Generating octave {octave + 1}/{self.octaves}")
            layer_seed = (
                settings.noise.seed + octave
                if settings.noise.seed is not None
                else None
            )

            layer = Perlin2D(self.scale, seed=layer_seed, shape=self.shape)()

            if self.turbulence:
                gamma = 0.5
                layer = (torch.abs(layer) ** gamma) * current_amp
            else:
                layer = layer * current_amp

            total_noise += layer

            current_amp *= self.persistence
            current_scale *= self.lacunarity

        return total_noise


@dataclass
class WhiteNoise2D:
    width: int
    height: int

    def __call__(self) -> torch.Tensor:
        """
        Generate 2D white noise.

        Returns:
            Noise tensor of shape (height, width) with values in [0, 1]
        """
        logging.debug("Generating white noise")
        return torch.empty((self.height, self.width), device=device).uniform_()
