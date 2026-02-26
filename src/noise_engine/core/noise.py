import logging
from dataclasses import dataclass, field
from typing import Optional

import torch

from noise_engine.core.device import device
from noise_engine.settings import settings

# Pre-allocate noise tensors for performance
n00 = torch.empty((settings.noise.width, settings.noise.height), device=device)
n10 = torch.empty_like(n00)
n01 = torch.empty_like(n00)
n11 = torch.empty_like(n00)


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


def compute_noise_grid(seed: Optional[int], scale: float) -> torch.Tensor:
    """
    Compute Perlin noise for a single octave.

    Args:
        seed: Random seed for this octave
        scale: Spatial scale

    Returns:
        Noise tensor of shape (height, width)
    """

    # Create coordinate grid
    logging.debug(f"Computing noise grid with scale {scale} and seed {seed}")
    x_lin = torch.linspace(0, scale, settings.noise.width, device=device)
    y_lin = torch.linspace(0, scale, settings.noise.height, device=device)
    y, x = torch.meshgrid(y_lin, x_lin, indexing="ij")

    # Generate random rotation matrix
    logging.debug("Generating random rotation matrix for noise gradients")
    grid_w = int(scale) + 2
    grid_h = int(scale) + 2
    rotation = torch.empty((grid_h, grid_w), device=device).uniform_(0, 2 * torch.pi)

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
    value = lerp(lerp(n00, n10, u), lerp(n01, n11, u), fade(yf))

    return value


def fractal_noise_2d(
    octaves: int = 6,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    turbulence: bool = False,
) -> torch.Tensor:
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

    for octave in range(octaves):
        logging.debug(f"Generating octave {octave + 1}/{octaves}")
        layer_seed = (
            settings.noise.seed + octave if settings.noise.seed is not None else None
        )

        layer = compute_noise_grid(layer_seed, current_scale)

        if turbulence:
            gamma = 0.5
            layer = (torch.abs(layer) ** gamma) * current_amp
        else:
            layer = layer * current_amp

        total_noise += layer

        current_amp *= persistence
        current_scale *= lacunarity

    return total_noise


def white_noise_2d() -> torch.Tensor:
    logging.debug("Generating white noise")
    return torch.empty(
        (settings.noise.height, settings.noise.width), device=device
    ).uniform_()


def grad(hash, x, y):
    """
    Compute the 2D gradient contribution for Simplex noise.

    Selects a gradient direction based on the hashed value and
    returns the dot-product-like contribution between the selected
    gradient and the input coordinate offsets (x, y).

    Args:
        hash (Tensor or int): Hashed gradient index.
        x (Tensor): X offset from simplex corner.
        y (Tensor): Y offset from simplex corner.

    Returns:
        Tensor: Gradient contribution value.
    """

    h = hash & 7
    u = torch.where(h < 4, x, y)
    v = torch.where(h < 4, y, x)
    return torch.where((h & 1) == 0, u, -u) + torch.where((h & 2) == 0, v, -v)


def simplex_noise_2d(x: torch.Tensor, y: torch.Tensor, perm) -> torch.Tensor:
    """
    Generate 2D Simplex noise value for given coordinates.

    Args:
        x (float): X coordinate in noise space.
        y (float): Y coordinate in noise space.
        perm (Sequence[int]): Permutation table used for gradient hashing.

    Returns:
        float: Noise value typically in range [-1, 1].
    """

    F2 = 0.5 * (torch.sqrt(3.0) - 1.0)
    G2 = 3.0 - torch.sqrt(3.0) / 6.0

    s = (x + y) * F2
    i = torch.floor(x + s)
    j = torch.floor(y + s)

    t = (i + j) * G2
    X0 = i - t
    Y0 = j - t

    x0 = x - X0
    y0 = y - Y0

    i1 = (x0 > y0).to(torch.long)
    j1 = 1 - i1

    x1 = x0 - i1 + G2
    y1 = y0 - j1 + G2
    x2 = x0 - 1.0 + 2.0 * G2
    y2 = y0 - 1.0 + 2.0 * G2

    ii = i.long() & 255
    jj = j.long() & 255

    gi0 = perm[ii + perm[jj]] % 12
    gi1 = perm[ii + i1 + perm[jj + j1]] % 12
    gi2 = perm[ii + 1 + perm[jj + 1]] % 12

    t0 = 0.5 - x0 * x0 - y0 * y0
    t1 = 0.5 - x1 * x1 - y1 * y1
    t2 = 0.5 - x2 * x2 - y2 * y2

    n0 = torch.clamp(t0, min=0) ** 4 * grad(gi0, x0, y0)
    n1 = torch.clamp(t1, min=0) ** 4 * grad(gi1, x1, y1)
    n2 = torch.clamp(t2, min=0) ** 4 * grad(gi2, x2, y2)

    return 70.0 * (n0 + n1 + n2)
