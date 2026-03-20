import torch

# Quintic fade curve coefficients (Horner's method)
FADE_C_4 = 6.0  # t^4 coefficient
FADE_C_3 = -15.0  # t^3 coefficient
FADE_C_2 = 10.0  # t^2 coefficient


def fade(t: torch.Tensor) -> torch.Tensor:
    """
    Compute the quintic fade curve for smooth interpolation.

    Uses Horner's method: t³ * (t² * 6 - 15*t + 10)
    Equivalent to: 6t⁵ - 15t⁴ + 10t³

    Args:
        t: Input tensor in range [0, 1]

    Returns:
        Smoothed tensor in range [0, 1] with zero first and second derivatives at endpoints
    """
    return t * t * (t * (t * (t * FADE_C_4 + FADE_C_3) + FADE_C_2))


def lerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Linear interpolation between two tensors.

    Uses PyTorch's optimized implementation for better performance.

    Args:
        a: Start tensor
        b: End tensor
        t: Interpolation factor in range [0, 1]

    Returns:
        Interpolated tensor: a + t * (b - a)
    """
    return torch.lerp(a, b, t)
