"""
PyTorch Noise Engine - A comprehensive library for generating various types of noise.

This package provides implementations of different noise algorithms including:
- Perlin noise
- Simplex noise
- White noise
- Fractal noise (fBm)

All noise generators are implemented using PyTorch for GPU acceleration and efficient computation.

Example:
    >>> import noise_engine as ne
    >>> # Generate 2D Perlin noise
    >>> perlin = ne.noise.Perlin2D(shape=(100, 100))
    >>> result = perlin()
    >>> print(result.shape)
    torch.Size([100, 100])
    
    >>> # Generate 3D Fractal noise
    >>> fractal = ne.noise.FractalNoise3D(shape=(50, 50, 50), octaves=4)
    >>> result = fractal()
    >>> print(result.shape)
    torch.Size([50, 50, 50])

Attributes:
    noise (Noise): Convenient namespace for accessing all noise generation classes.
"""

from noise_engine.core.noise import noise

__version__ = "0.1.0"
__author__ = "Complex-CN"
__all__ = ["noise"]
