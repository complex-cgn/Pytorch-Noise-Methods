import torch

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