"""
Noise Engine SDK - Main module for easy access to all noise generation methods.
"""

from noise_engine.core.noise_types.perlin import (
    PerlinNoise1D,
    PerlinNoise2D,
    PerlinNoise3D,
)
from noise_engine.core.noise_types.simplex import (
    SimplexNoise1D,
    SimplexNoise2D,
    SimplexNoise3D,
)
from noise_engine.core.noise_types.white import (
    WhiteNoise1D,
    WhiteNoise2D,
    WhiteNoise3D,
)
from noise_engine.core.noise_types.fractal import (
    FractalNoise1D,
    FractalNoise2D,
    FractalNoise3D,
)

# Export all noise classes for easy access
__all__ = [
    "PerlinNoise1D",
    "PerlinNoise2D",
    "PerlinNoise3D",
    "SimplexNoise1D",
    "SimplexNoise2D",
    "SimplexNoise3D",
    "WhiteNoise1D",
    "WhiteNoise2D",
    "WhiteNoise3D",
    "FractalNoise1D",
    "FractalNoise2D",
    "FractalNoise3D",
]


# Create a convenient noise namespace
class Noise:
    """Convenient access to all noise generation methods.

    This class provides a clean namespace for accessing all noise generation classes.
    It groups noise types by algorithm (Perlin, Simplex, White, Fractal) and dimensionality (1D, 2D, 3D).

    Example:
        >>> from noise_engine import noise
        >>> # Create 2D Perlin noise
        >>> perlin_noise = noise.Perlin2D(shape=(100, 100))
        >>> # Create 3D Fractal noise
        >>> fractal_noise = noise.FractalNoise3D(shape=(50, 50, 50), octaves=4)
        >>> # Generate the noise
        >>> result = perlin_noise()
        >>> print(result.shape)
        torch.Size([100, 100])

    Attributes:
        Perlin1D (class): 1D Perlin noise generator.
        Perlin2D (class): 2D Perlin noise generator.
        Perlin3D (class): 3D Perlin noise generator.
        Simplex1D (class): 1D Simplex noise generator.
        Simplex2D (class): 2D Simplex noise generator.
        Simplex3D (class): 3D Simplex noise generator.
        WhiteNoise1D (class): 1D White noise generator.
        WhiteNoise2D (class): 2D White noise generator.
        WhiteNoise3D (class): 3D White noise generator.
        FractalNoise1D (class): 1D Fractal noise generator.
        FractalNoise2D (class): 2D Fractal noise generator.
        FractalNoise3D (class): 3D Fractal noise generator.
    """

    # Perlin noise
    Perlin1D = PerlinNoise1D
    Perlin2D = PerlinNoise2D
    Perlin3D = PerlinNoise3D

    # Simplex noise
    Simplex1D = SimplexNoise1D
    Simplex2D = SimplexNoise2D
    Simplex3D = SimplexNoise3D

    # White noise
    WhiteNoise1D = WhiteNoise1D
    WhiteNoise2D = WhiteNoise2D
    WhiteNoise3D = WhiteNoise3D

    # Fractal noise (fBm)
    FractalNoise1D = FractalNoise1D
    FractalNoise2D = FractalNoise2D
    FractalNoise3D = FractalNoise3D


# Create a global instance for easy access
noise = Noise()
