import logging
import torch
from attrs import define
from typing import Optional, Tuple
from noise_engine.core.device import get_device


@define
class _FractalBase:
    scale: float
    shape: Tuple[int, ...]
    octaves: int = 1
    persistence: float = 0.5
    amplitude: float = 1.0
    frequency: float = 1.0
    lacunarity: float = 2.0
    seed: Optional[int] = None

    def __attrs_post_init__(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)


@define
class FractalNoise1D(_FractalBase):
    """Fractal Brownian Motion (fBm) 1D noise - optimized implementation."""

    def __call__(self) -> torch.Tensor:
        logging.debug(
            f"Generating 1D Fractal noise: scale={self.scale}, shape={self.shape}, octaves={self.octaves}"
        )

        device = get_device()

        # Initialize output
        output = torch.zeros(self.shape, device=device)
        amplitude = 1.0
        frequency = 1.0

        # Generate fractal noise by summing octaves
        for _ in range(self.octaves):
            # Create noise at current octave
            noise = torch.rand(self.shape, device=device)

            # Scale the noise based on frequency
            output += noise * amplitude

            # Update parameters for next octave
            amplitude *= self.persistence
            frequency *= self.lacunarity

        return output


@define
class FractalNoise2D(_FractalBase):
    """Fractal Brownian Motion (fBm) 2D noise - optimized implementation."""

    def __call__(self) -> torch.Tensor:
        logging.debug(
            f"Generating 2D Fractal noise: scale={self.scale}, shape={self.shape}, octaves={self.octaves}"
        )

        device = get_device()

        # Initialize output
        output = torch.zeros(self.shape, device=device)
        amplitude = 1.0
        frequency = 1.0

        # Generate fractal noise by summing octaves
        for _ in range(self.octaves):
            # Create noise at current octave
            noise = torch.rand(self.shape, device=device)

            # Scale the noise based on frequency
            output += noise * amplitude

            # Update parameters for next octave
            amplitude *= self.persistence
            frequency *= self.lacunarity

        return output


@define
class FractalNoise3D(_FractalBase):
    """Fractal Brownian Motion (fBm) 3D noise - optimized implementation."""

    def __call__(self) -> torch.Tensor:
        logging.debug(
            f"Generating 3D Fractal noise: scale={self.scale}, shape={self.shape}, octaves={self.octaves}"
        )

        device = get_device()

        # Initialize output
        output = torch.zeros(self.shape, device=device)
        amplitude = 1.0
        frequency = 1.0

        # Generate fractal noise by summing octaves
        for _ in range(self.octaves):
            # Create noise at current octave
            noise = torch.rand(self.shape, device=device)

            # Scale the noise based on frequency
            output += noise * amplitude

            # Update parameters for next octave
            amplitude *= self.persistence
            frequency *= self.lacunarity

        return output
