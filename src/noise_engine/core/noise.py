"""
Noise Engine SDK - Main module for easy access to all noise generation methods.
"""

<<<<<<< HEAD
from noise_engine.core.noise_opt.perlin_noise import (
=======
from noise_engine.core.noise_types.perlin import (
>>>>>>> e5d8bbe (feat: update noise generation to fractal noise with configurable octaves and version bump to 0.2.0)
    PerlinNoise1D,
    PerlinNoise2D,
    PerlinNoise3D,
)
<<<<<<< HEAD
from noise_engine.core.noise_opt.simplex_noise import (
=======
from noise_engine.core.noise_types.simplex import (
>>>>>>> e5d8bbe (feat: update noise generation to fractal noise with configurable octaves and version bump to 0.2.0)
    SimplexNoise1D,
    SimplexNoise2D,
    SimplexNoise3D,
)
<<<<<<< HEAD
from noise_engine.core.noise_opt.white_noise import (
=======
from noise_engine.core.noise_types.white import (
>>>>>>> e5d8bbe (feat: update noise generation to fractal noise with configurable octaves and version bump to 0.2.0)
    WhiteNoise1D,
    WhiteNoise2D,
    WhiteNoise3D,
)
<<<<<<< HEAD
from noise_engine.core.noise_opt.fractal_noise import (
=======
from noise_engine.core.noise_types.fractal import (
>>>>>>> e5d8bbe (feat: update noise generation to fractal noise with configurable octaves and version bump to 0.2.0)
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
