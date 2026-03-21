import logging
from typing import Optional, Tuple
import torch
from attrs import define
from noise_engine.core.device import get_device
from noise_engine.core.noise_opt.perlin_noise import (
    PerlinNoise1D,
    PerlinNoise2D,
    PerlinNoise3D,
)


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
            layer = PerlinNoise1D(
                scale=current_scale, shape=self.shape, seed=layer_seed
            )()

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
            layer = PerlinNoise2D(
                scale=current_scale, shape=self.shape, seed=layer_seed
            )()

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
            layer = PerlinNoise3D(
                scale=current_scale, shape=self.shape, seed=layer_seed
            )()

            if self.turbulence:
                layer = torch.abs(layer) ** self.turbulence_gamma

            total += layer * current_amp
            current_amp *= self.persistence
            current_scale *= self.lacunarity

        return total
