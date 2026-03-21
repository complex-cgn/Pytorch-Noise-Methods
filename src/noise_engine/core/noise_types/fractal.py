import logging
import torch
from attrs import define
from typing import Optional, Tuple
from noise_engine.core.device import get_device


@define
class _FractalBase:
    """Base class for fractal noise generation with configurable parameters.

    This class provides common parameters for generating fractal noise using the
    fractional Brownian motion (fBm) algorithm. It serves as a foundation for
    1D, 2D, and 3D fractal noise implementations.

    Args:
        scale (float): The scaling factor for the noise output.
        shape (Tuple[int, ...]): The output tensor shape.
        octaves (int): The number of noise layers to combine. More octaves
            create more detailed noise but increase computation time.
        persistence (float): The amplitude multiplier between successive octaves.
            Values closer to 1.0 produce rougher noise, while values closer to 0.0
            produce smoother noise.
        amplitude (float): The initial amplitude of the noise.
        frequency (float): The initial frequency of the noise.
        lacunarity (float): The frequency multiplier between successive octaves.
            Common values are 2.0 for standard fractal noise.
        seed (Optional[int]): Random seed for reproducible results. If None,
            results will vary between calls.

    Attributes:
        scale (float): The scaling factor for the noise output.
        shape (Tuple[int, ...]): The output tensor shape.
        octaves (int): The number of noise layers to combine. More octaves
            create more detailed noise but increase computation time.
        persistence (float): The amplitude multiplier between successive octaves.
            Values closer to 1.0 produce rougher noise, while values closer to 0.0
            produce smoother noise.
        amplitude (float): The initial amplitude of the noise.
        frequency (float): The initial frequency of the noise.
        lacunarity (float): The frequency multiplier between successive octaves.
            Common values are 2.0 for standard fractal noise.
        seed (Optional[int]): Random seed for reproducible results. If None,
            results will vary between calls.
    """

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
    """Generate 1D Fractal Brownian Motion (fBm) noise.

    This class generates 1D fractal noise using the fractional Brownian motion
    algorithm by combining multiple octaves of noise with varying frequencies
    and amplitudes.

    The resulting noise will have a self-similar, natural-looking pattern that
    can be used for various applications like terrain generation, texture synthesis,
    or procedural content generation.

    Returns:
        torch.Tensor: A 1D tensor of shape specified by the 'shape' parameter,
            containing the generated fractal noise values in the range [0, 1].

    Example:
        >>> noise_gen = FractalNoise1D(shape=(100,), octaves=4, persistence=0.5)
        >>> noise = noise_gen()
        >>> print(noise.shape)
        torch.Size([100])
    """

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
    """Generate 2D Fractal Brownian Motion (fBm) noise.

    This class generates 2D fractal noise using the fractional Brownian motion
    algorithm by combining multiple octaves of noise with varying frequencies
    and amplitudes.

    The resulting noise will have a self-similar, natural-looking pattern that
    can be used for various applications like terrain generation, texture synthesis,
    or procedural content generation.

    Returns:
        torch.Tensor: A 2D tensor of shape specified by the 'shape' parameter,
            containing the generated fractal noise values in the range [0, 1].

    Example:
        >>> noise_gen = FractalNoise2D(shape=(100, 100), octaves=4, persistence=0.5)
        >>> noise = noise_gen()
        >>> print(noise.shape)
        torch.Size([100, 100])
    """

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
    """Generate 3D Fractal Brownian Motion (fBm) noise.

    This class generates 3D fractal noise using the fractional Brownian motion
    algorithm by combining multiple octaves of noise with varying frequencies
    and amplitudes.

    The resulting noise will have a self-similar, natural-looking pattern that
    can be used for various applications like terrain generation, texture synthesis,
    or procedural content generation in 3D space.

    Returns:
        torch.Tensor: A 3D tensor of shape specified by the 'shape' parameter,
            containing the generated fractal noise values in the range [0, 1].

    Example:
        >>> noise_gen = FractalNoise3D(shape=(50, 50, 50), octaves=4, persistence=0.5)
        >>> noise = noise_gen()
        >>> print(noise.shape)
        torch.Size([50, 50, 50])
    """

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
