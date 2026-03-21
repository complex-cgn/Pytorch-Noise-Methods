"""
Main noise generator module for the Noise Engine.
This module imports and re-exports all noise classes for easy access.
"""

# Import all noise classes from their respective modules
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
