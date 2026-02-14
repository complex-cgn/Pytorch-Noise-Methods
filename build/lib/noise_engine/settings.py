from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


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
    noise_options: NoiseSettings
    render_options: RenderSettings


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "settings.yaml"

with open(CONFIG_PATH) as f:
    data = yaml.safe_load(f)

settings = Settings(**data)
