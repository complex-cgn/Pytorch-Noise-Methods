import logging
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, ValidationError

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "settings.yaml"


class NoiseSettings(BaseModel):
    width: int
    height: int
    scale: float
    num_octaves: int
    seed: Optional[int] = None


class RenderSettings(BaseModel):
    color_map: str
    export_dpi: int
    show_plot: bool
    output_path: str


class Settings(BaseModel):
    noise: NoiseSettings = Field(alias="noise_options")
    render: RenderSettings = Field(alias="render_options")

    @classmethod
    def load_from_yaml(cls, path: Path) -> "Settings":

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


try:
    settings = Settings.load_from_yaml(CONFIG_PATH)
    print(f"Settings Loaded: {settings.render.output_path}")
except Exception as e:
    print(f"The system could not be started: {e}")
