"""
Noise Engine SDK - Main module for easy access to all noise generation methods.
"""

from noise_engine.core.noise_opt.perlin_noise import (
    PerlinNoise1D,
    PerlinNoise2D,
    PerlinNoise3D,
)
from noise_engine.core.noise_opt.simplex_noise import (
    SimplexNoise1D,
    SimplexNoise2D,
    SimplexNoise3D,
)
from noise_engine.core.noise_opt.white_noise import (
    WhiteNoise1D,
    WhiteNoise2D,
    WhiteNoise3D,
)
from noise_engine.core.noise_opt.fractal_noise import (
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
    """Convenient access to all noise generation methods."""

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
