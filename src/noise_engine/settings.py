"""
Settings module for the Noise Engine.
This module handles configuration loading and validation.

Example:
    >>> from noise_engine.settings import get_settings
    >>> settings = get_settings()
    >>> print(settings.noise.scale)
    1.0

Attributes:
    CONFIG_PATH (Path): Path to the settings configuration file.
"""

import logging
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, ValidationError

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "settings.yaml"


class NoiseSettings(BaseModel):
    """Configuration settings for noise generation.

    Attributes:
        width (int): Width of the generated noise.
        height (int): Height of the generated noise.
        scale (float): Scale factor for the noise.
        num_octaves (int): Number of octaves for fractal noise.
        seed (Optional[int]): Random seed for reproducible results.
    """
    width: int
    height: int
    scale: float
    num_octaves: int
    seed: Optional[int] = None


class RenderSettings(BaseModel):
    """Configuration settings for rendering.

    Attributes:
        color_map (str): Matplotlib color map to use for visualization.
        export_dpi (int): DPI for exported plots.
        show_plot (bool): Whether to display the plot.
        output_path (str): Path where plots will be saved.
    """
    color_map: str
    export_dpi: int
    show_plot: bool
    output_path: str


class Settings(BaseModel):
    """Main settings configuration.

    Attributes:
        noise (NoiseSettings): Noise generation settings.
        render (RenderSettings): Rendering settings.
    """
    noise: NoiseSettings = Field(alias="noise_options")
    render: RenderSettings = Field(alias="render_options")

    @classmethod
    def load_from_yaml(cls, path: Path) -> "Settings":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Parsed Settings instance.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            ValidationError: If configuration is invalid.
        """
        if not path.exists():
            logging.error(f"Configuration file not found: {path}")
            raise FileNotFoundError(f"Configuration file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
                return cls(**config_data)
        except (yaml.YAMLError, ValidationError) as e:
            logging.error("Configuration error: %s", e)
            raise


# Lazy singleton instance
_settings_instance: Settings | None = None


def get_settings(path: Path = CONFIG_PATH) -> Settings:
    """Get settings instance (cached after first call).

    Args:
        path: Configuration file path. Defaults to bundled settings.yaml.

    Returns:
        Configured Settings instance.

    Raises:
        FileNotFoundError: If configuration file doesn't exist.
        ValidationError: If configuration is invalid.
    """
    global _settings_instance

    # Return cached instance if already loaded
    if _settings_instance is not None:
        return _settings_instance

    # Load new instance
    _settings_instance = Settings.load_from_yaml(path)
    logging.debug(f"Settings loaded from {path}")
    return _settings_instance


def reset_settings() -> None:
    """Reset cached settings (useful for testing)."""
    global _settings_instance
    _settings_instance = None
