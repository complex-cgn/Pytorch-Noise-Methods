import logging
from typing import Optional, Tuple
import torch
from attrs import define, field, validators
from noise_engine.core.device import get_device

"""
TODO:
>>> Ridged/Billow Noise
>>> Curl Noise
>>> Worley Noise
>>> 3D Terrain Generator
"""

# fmt: off
# Permutation table (Ken Perlin's original 256-entry table)
_PERM_TABLE = [
    151,160,137, 91, 90, 15,131, 13,201, 95, 96, 53,194,233,  7,225,
    140, 36,103, 30, 69,142,  8, 99, 37,240, 21, 10, 23,190,  6,148,
    247,120,234, 75,  0, 26,197, 62, 94,252,219,203,117, 35, 11, 32,
     57,177, 33, 88,237,149, 56, 87,174, 20,125,136,171,168, 68,175,
     74,165, 71,134,139, 48, 27,166, 77,146,158,231, 83,111,229,122,
     60,211,133,230,220,105, 92, 41, 55, 46,245, 40,244,102,143, 54,
     65, 25, 63,161,  1,216, 80, 73,209, 76,132,187,208, 89, 18,169,
    200,196,135,130,116,188,159, 86,164,100,109,198,173,186,  3, 64,
     52,217,226,250,124,123,  5,202, 38,147,118,126,255, 82, 85,212,
    207,206, 59,227, 47, 16, 58, 17,182,189, 28, 42,223,183,170,213,
    119,248,152,  2, 44,154,163, 70,221,153,101,155,167, 43,172,  9,
    129, 22, 39,253, 19, 98,108,110, 79,113,224,232,178,185,112,104,
    218,246, 97,228,251, 34,242,193,238,210,144, 12,191,179,162,241,
     81, 51,145,235,249, 14,239,107, 49,192,214, 31,181,199,106,157,
    184, 84,204,176,115,121, 50, 45,127,  4,150,254,138,236,205, 93,
    222,114, 67, 29, 24, 72,243,141,128,195, 78, 66,215, 61,156,180,
]
# fmt: on


def _build_perm(seed: int | None, device: torch.device) -> torch.Tensor:
    """Build a 512-entry permutation table, optionally shuffled by seed."""
    perm = torch.tensor(_PERM_TABLE, dtype=torch.long, device=device)
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        idx = torch.randperm(256, generator=g, device=device)
        perm = perm[idx]
    # Double the table so we never need modulo in hot path
    return torch.cat([perm, perm])  # shape (512,)


# ---------------------------------------------------------------------------
# Gradient tables (pre-computed, stored as tensors)
# ---------------------------------------------------------------------------

# 1D: gradients are just ±1
_GRAD1 = torch.tensor([1.0, -1.0])

# 2D: 8 gradient directions (corners + edges of unit square, normalised)
_GRAD2 = torch.tensor(
    [
        [1.0, 1.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [-1.0, -1.0],
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
    ],
    dtype=torch.float32,
)

# 3D: 12 gradient directions (mid-points of a cube's edges)
_GRAD3 = torch.tensor(
    [
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
        [-1.0, -1.0, 0.0],
        [1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 0.0, -1.0],
        [0.0, 1.0, 1.0],
        [0.0, -1.0, 1.0],
        [0.0, 1.0, -1.0],
        [0.0, -1.0, -1.0],
    ],
    dtype=torch.float32,
)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


@define
class _SimplexBase:
    """Shared config for all simplex noise generators."""

    scale: float = field(default=1.0, validator=validators.gt(0))
    seed: int | None = field(default=None)

    def _perm(self, device: torch.device) -> torch.Tensor:
        """Get permutation table for given device."""
        return _build_perm(self.seed, device)

    def _grad1(self, device: torch.device) -> torch.Tensor:
        """Get 1D gradient table for given device."""
        return _GRAD1.to(device)

    def _grad2(self, device: torch.device) -> torch.Tensor:
        """Get 2D gradient table for given device"""
        return _GRAD2.to(device)

    def _grad3(self, device: torch.device) -> torch.Tensor:
        """Get 3D gradient table for given device."""
        return _GRAD3.to(device)


@define
class SimplexNoise1D(_SimplexBase):
    """
    1D Simplex noise.

    Returns a tensor of shape `(n,)` with values in approximately [-1, 1].

    Parameters
    ----------
    n     : number of output samples
    scale : wavelength in samples (larger → smoother)
    seed  : optional RNG seed for reproducibility
    """

    n: int = field(default=256, validator=validators.gt(0))

    def __call__(self) -> torch.Tensor:
        logging.debug(f"Simplex1D: n={self.n}, scale={self.scale}")
        device = get_device()
        perm = self._perm(device)
        grad1 = self._grad1(device)

        # Sample coordinates
        x = torch.arange(self.n, dtype=torch.float32, device=device) / self.scale

        # Integer cell and fractional offset
        i0 = x.floor().long()
        x0 = x - i0.float()
        x1 = x0 - 1.0

        # Permuted gradient indices
        gi0 = perm[i0 & 255] & 1  # mod 2 → index into grad1
        gi1 = perm[(i0 + 1) & 255] & 1

        # Gradient lookup
        g0 = grad1[gi0]  # (n,)
        g1 = grad1[gi1]  # (n,)

        # Kernel: t = 1 - x², clamped to [0, 1], then t⁴
        t0 = (1.0 - x0 * x0).clamp(min=0.0) ** 4
        t1 = (1.0 - x1 * x1).clamp(min=0.0) ** 4

        # Contribution from each corner
        n0 = t0 * (g0 * x0)
        n1 = t1 * (g1 * x1)

        # Scale to approx [-1, 1]
        return (n0 + n1) * 0.395 * 2.0


# Skewing constants for 2D
_F2 = 0.5 * (3.0**0.5 - 1.0)  # skew
_G2 = (3.0 - 3.0**0.5) / 6.0  # unskew


@define
class SimplexNoise2D(_SimplexBase):
    """
    2D Simplex noise.

    Returns a tensor of shape `(H, W)` with values in approximately [-1, 1].

    Parameters
    ----------
    shape : (H, W) output resolution
    scale : wavelength in pixels (larger → smoother)
    seed  : optional RNG seed for reproducibility
    """

    shape: tuple[int, int] = field(default=(256, 256))

    @shape.validator
    def _check_shape(self, attribute, value):
        if len(value) != 2 or any(v <= 0 for v in value):
            raise ValueError("shape must be (H, W) with positive integers")

    def __call__(self) -> torch.Tensor:
        H, W = self.shape
        logging.debug(f"Simplex2D: shape={self.shape}, scale={self.scale}")
        device = get_device()
        perm = self._perm(device)
        grad2 = self._grad2(device)

        # Build coordinate grid
        ys = torch.arange(H, dtype=torch.float32, device=device) / self.scale
        xs = torch.arange(W, dtype=torch.float32, device=device) / self.scale
        y, x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W) each

        # --------------- Skew input space to determine simplex cell --------
        s = (x + y) * _F2
        i = (x + s).floor().long()  # cell coords
        j = (y + s).floor().long()

        t = (i + j).float() * _G2
        # Unskew back — corner 0 in original space
        X0 = i.float() - t
        Y0 = j.float() - t
        x0 = x - X0
        y0 = y - Y0

        # Determine which simplex triangle we're in
        # i1, j1 = second corner offset (either (1,0) or (0,1))
        i1 = (x0 >= y0).long()
        j1 = 1 - i1

        # Offsets for corners 1 and 2
        x1 = x0 - i1.float() + _G2
        y1 = y0 - j1.float() + _G2
        x2 = x0 - 1.0 + 2.0 * _G2
        y2 = y0 - 1.0 + 2.0 * _G2

        # --------------- Gradient index lookup ----------------------------
        ii = i & 255
        jj = j & 255

        gi0 = perm[ii + perm[jj]] % 8
        gi1 = perm[ii + i1 + perm[jj + j1]] % 8
        gi2 = perm[ii + 1 + perm[jj + 1]] % 8

        g0 = grad2[gi0]  # (H, W, 2)
        g1 = grad2[gi1]
        g2 = grad2[gi2]

        # --------------- Kernel & contributions ---------------------------
        n0 = (0.5 - x0 * x0 - y0 * y0).clamp(min=0.0) ** 4 * (
            g0[..., 0] * x0 + g0[..., 1] * y0
        )
        n1 = (0.5 - x1 * x1 - y1 * y1).clamp(min=0.0) ** 4 * (
            g1[..., 0] * x1 + g1[..., 1] * y1
        )
        n2 = (0.5 - x2 * x2 - y2 * y2).clamp(min=0.0) ** 4 * (
            g2[..., 0] * x2 + g2[..., 1] * y2
        )

        # Scale to approx [-1, 1]
        return (n0 + n1 + n2) * 70.0


# Skewing constants for 3D
_F3 = 1.0 / 3.0
_G3 = 1.0 / 6.0


@define
class SimplexNoise3D(_SimplexBase):
    """
    3D Simplex noise.

    Returns a tensor of shape `(D, H, W)` with values in approximately [-1, 1].

    Parameters
    ----------
    shape : (D, H, W) output resolution
    scale : wavelength in voxels (larger → smoother)
    seed  : optional RNG seed for reproducibility
    """

    shape: tuple[int, int, int] = field(default=(64, 64, 64))

    @shape.validator
    def _check_shape(self, attribute, value):
        if len(value) != 3 or any(v <= 0 for v in value):
            raise ValueError("shape must be (D, H, W) with positive integers")

    def __call__(self) -> torch.Tensor:
        D, H, W = self.shape
        logging.debug(f"Simplex3D: shape={self.shape}, scale={self.scale}")
        device = get_device()
        perm = self._perm(device)
        grad3 = self._grad3(device)  # (12, 3)

        # Coordinate grid
        zs = torch.arange(D, dtype=torch.float32, device=device) / self.scale
        ys = torch.arange(H, dtype=torch.float32, device=device) / self.scale
        xs = torch.arange(W, dtype=torch.float32, device=device) / self.scale
        z, y, x = torch.meshgrid(zs, ys, xs, indexing="ij")  # (D, H, W) each

        # --------------- Skew -----------------------------------------------
        s = (x + y + z) * _F3
        i = (x + s).floor().long()
        j = (y + s).floor().long()
        k = (z + s).floor().long()

        t = (i + j + k).float() * _G3
        X0 = i.float() - t
        Y0 = j.float() - t
        Z0 = k.float() - t
        x0 = x - X0
        y0 = y - Y0
        z0 = z - Z0

        # --------------- Simplex ordering -----------------------------------
        # Determine which of the 6 tetrahedral simplices we're in
        x_ge_y = x0 >= y0
        y_ge_z = y0 >= z0
        x_ge_z = x0 >= z0

        # Corner 1
        i1 = (x_ge_y & x_ge_z).long()
        j1 = (~x_ge_y & y_ge_z).long()
        k1 = (~x_ge_z & ~y_ge_z).long()

        # Corner 2
        i2 = (x_ge_y | x_ge_z).long()
        j2 = (~x_ge_y | y_ge_z).long()
        k2 = (~(x_ge_z & y_ge_z)).long()

        # Offsets for corners 1, 2, 3
        x1 = x0 - i1.float() + _G3
        y1 = y0 - j1.float() + _G3
        z1 = z0 - k1.float() + _G3

        x2 = x0 - i2.float() + 2.0 * _G3
        y2 = y0 - j2.float() + 2.0 * _G3
        z2 = z0 - k2.float() + 2.0 * _G3

        x3 = x0 - 1.0 + 3.0 * _G3
        y3 = y0 - 1.0 + 3.0 * _G3
        z3 = z0 - 1.0 + 3.0 * _G3

        # --------------- Gradient indices -----------------------------------
        ii = i & 255
        jj = j & 255
        kk = k & 255

        gi0 = perm[ii + perm[jj + perm[kk]]] % 12
        gi1 = perm[ii + i1 + perm[jj + j1 + perm[kk + k1]]] % 12
        gi2 = perm[ii + i2 + perm[jj + j2 + perm[kk + k2]]] % 12
        gi3 = perm[ii + 1 + perm[jj + 1 + perm[kk + 1]]] % 12

        g0 = grad3[gi0]  # (D, H, W, 3)
        g1 = grad3[gi1]
        g2 = grad3[gi2]
        g3 = grad3[gi3]

        # --------------- Contributions --------------------------------------
        def corner(g, dx, dy, dz):
            t_ = (0.6 - dx * dx - dy * dy - dz * dz).clamp(min=0.0) ** 4
            dot = g[..., 0] * dx + g[..., 1] * dy + g[..., 2] * dz
            return t_ * dot

        n0 = corner(g0, x0, y0, z0)
        n1 = corner(g1, x1, y1, z1)
        n2 = corner(g2, x2, y2, z2)
        n3 = corner(g3, x3, y3, z3)

        # Scale to approx [-1, 1]
        return (n0 + n1 + n2 + n3) * 32.0
