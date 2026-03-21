import torch
from contextlib import contextmanager


def get_default_device() -> torch.device:
    """Get the default device (CUDA if available, otherwise CPU).

    Returns:
        torch.device: Default device object.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device(device_str: str | torch.device = None) -> torch.device:
    """Get and validate a PyTorch device.

    Args:
        device_str (str | torch.device, optional): Device string ('cpu', 'cuda', 'cuda:0', etc.) or torch.device object.
            Defaults to CUDA if available, otherwise CPU.

    Returns:
        torch.device: Validated torch.device object.

    Raises:
        ValueError: If CUDA is requested but not available.

    Example:
        >>> device = get_device("cuda")
        >>> print(device)
        cuda:0
    """
    if device_str is None:
        return get_default_device()

    if isinstance(device_str, torch.device):
        return device_str

    device = torch.device(device_str)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA not available. Please use 'cpu' or omit device argument for auto-detection."
        )

    return device


@contextmanager
def use_device(device_str: str | torch.device):
    """Context manager for temporarily switching devices.

    Args:
        device_str (str | torch.device): Target device string or torch.device object.

    Yields:
        torch.device: The validated device object.

    Example:
        >>> with use_device("cpu") as dev:
        ...     tensor = torch.randn(100, device=dev)
    """
    device = get_device(device_str)
    yield device
