import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class Noise:
    """
    A class for generating Perlin noise using PyTorch GPU acceleration.

    Attributes:
        width: Width of the noise grid
        height: Height of the noise grid
        scale: Spatial scale of the noise
        seed: Random seed for reproducibility
        device: Device to use for computation (cuda/cpu)
    """

    width: int
    height: int
    scale: float = 4.0
    seed: Optional[int] = None
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )

    # Pre-allocated noise values for performance
    _n00: torch.Tensor = field(init=False)
    _n10: torch.Tensor = field(init=False)
    _n01: torch.Tensor = field(init=False)
    _n11: torch.Tensor = field(init=False)

    def __post_init__(self):
        """Initialize pre-allocated noise tensors."""
        self._n00 = torch.empty((self.height, self.width), device=self.device)
        self._n10 = torch.empty_like(self._n00)
        self._n01 = torch.empty_like(self._n00)
        self._n11 = torch.empty_like(self._n00)

        if self.seed is not None:
            torch.manual_seed(self.seed)

    @staticmethod
    def _fade(t: torch.Tensor) -> torch.Tensor:
        """
        Compute the fade curve for smooth interpolation.

        Uses the quintic polynomial: t³ * (t * (6*t - 15) + 10)

        Args:
            t: Input tensor in range [0, 1]

        Returns:
            Smoothed tensor in range [0, 1]
        """
        return t * t * t * (t * (t * 6 - 15) + 10)

    @staticmethod
    def _lerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
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

    def _compute_noise_grid(self, seed: Optional[int], scale: float) -> torch.Tensor:
        """
        Compute Perlin noise for a single octave.

        Args:
            seed: Random seed for this octave
            scale: Spatial scale

        Returns:
            Noise tensor of shape (height, width)
        """
        # Get pre-allocated tensors
        n00, n10, n01, n11 = self._n00, self._n10, self._n01, self._n11

        # Create coordinate grid
        logging.debug(f"Computing noise grid with scale {scale} and seed {seed}")
        x_lin = torch.linspace(0, scale, self.width, device=self.device)
        y_lin = torch.linspace(0, scale, self.height, device=self.device)
        y, x = torch.meshgrid(y_lin, x_lin, indexing="ij")

        # Generate random rotation matrix
        logging.debug("Generating random rotation matrix for noise gradients")
        grid_w = int(scale) + 2
        grid_h = int(scale) + 2
        rotation = torch.empty((grid_h, grid_w), device=self.device).uniform_(
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
        u = self._fade(xf)
        value = self._lerp(
            self._lerp(n00, n10, u), self._lerp(n01, n11, u), self._fade(yf)
        )

        return value

    def fractal_noise_2d(
        self,
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
        total_noise = torch.zeros((self.height, self.width), device=self.device)
        current_scale = self.scale
        current_amp = 0.1

        for octave in range(octaves):
            logging.debug(f"Generating octave {octave + 1}/{octaves}")
            layer_seed = self.seed + octave if self.seed is not None else None

            layer = self._compute_noise_grid(layer_seed, current_scale)

            if turbulence:
                gamma = 0.5
                layer = (torch.abs(layer) ** gamma) * current_amp
            else:
                layer = layer * current_amp

            total_noise += layer

            current_amp *= persistence
            current_scale *= lacunarity

        return total_noise

    def white_noise_2d(self):
        logging.debug("Generating white noise")
        return torch.empty((self.height, self.width), device=self.device).uniform_()

    @staticmethod
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

    def simplex_noise_2d(self, x: torch.Tensor, y: torch.Tensor, perm) -> torch.Tensor:
        """
        Generate 2D Simplex noise value for given coordinates.

        Args:
            x (float): X coordinate in noise space.
            y (float): Y coordinate in noise space.
            perm (Sequence[int]): Permutation table used for gradient hashing.

        Returns:
            float: Noise value typically in range [-1, 1].
        """
        f2 = 0.5 * (math.sqrt(3.0) - 1.0)
        g2 = (3.0 - math.sqrt(3.0)) / 6.0

        # Skew transformation for simplex grid
        logging.debug("Applying skew transformation for simplex grid")
        shift_factor = (x + y) * f2
        grid_i = torch.floor(x + shift_factor)
        grid_j = torch.floor(y + shift_factor)

        # Unskew operation to get actual coordinates
        logging.debug(
            "Applying unskew transformation to get simplex corner coordinates"
        )
        offset = (grid_i + grid_j) * g2
        simplex_corner_x = grid_i - offset
        simplex_corner_y = grid_j - offset

        # Fractional distances from simplex corner
        logging.debug("Calculating fractional offsets from simplex corner")
        frac_x = x - simplex_corner_x
        frac_y = y - simplex_corner_y

        # Identify which simplex corner is closer
        logging.debug("Determining simplex corner ordering")
        i1 = (simplex_corner_x > simplex_corner_y).int()
        j1 = 1 - i1

        # Compute coordinates relative to the three relevant simplex corners
        logging.debug("Computing coordinates relative to simplex corners")
        x0, y0 = simplex_corner_x, simplex_corner_y
        x1 = simplex_corner_x - i1 + g2
        y1 = simplex_corner_y - j1 + g2
        x2 = simplex_corner_x - 1.0 + 2.0 * g2
        y2 = simplex_corner_y - 1.0 + 2.0 * g2

        # Hash gradient indices using permutation table
        logging.debug("Hashing gradient indices using permutation table")
        ii = grid_i.long() & 255
        jj = grid_j.long() & 255

        grad_idx_0 = perm[ii + perm[jj]]
        grad_idx_1 = perm[ii + i1 + perm[jj + j1]]
        grad_idx_2 = perm[ii + 1 + perm[jj + 1]]

        # Compute kernel values (radial basis function)
        logging.debug("Computing kernel values for simplex corners")
        kernel_val_0 = 0.5 - x0 * x0 - y0 * y0
        kernel_val_1 = 0.5 - x1 * x1 - y1 * y1
        kernel_val_2 = 0.5 - x2 * x2 - y2 * y2

        # Gradient contributions from each simplex corner
        logging.debug("Calculating gradient contributions from simplex corners")
        contrib_0 = torch.where(
            kernel_val_0 < 0, 0.0, (kernel_val_0**4) * self.grad(grad_idx_0, x0, y0)
        )
        contrib_1 = torch.where(
            kernel_val_1 < 0, 0.0, (kernel_val_1**4) * self.grad(grad_idx_1, x1, y1)
        )
        contrib_2 = torch.where(
            kernel_val_2 < 0, 0.0, (kernel_val_2**4) * self.grad(grad_idx_2, x2, y2)
        )

        return 70.0 * (contrib_0 + contrib_1 + contrib_2)
