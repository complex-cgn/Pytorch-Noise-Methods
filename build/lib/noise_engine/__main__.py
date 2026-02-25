import logging

import click
import torch
from noise_engine.core.noise import Noise
from noise_engine.settings import settings
from noise_engine.utils import tensor
from noise_engine.utils.timer import Timer
from rich.logging import RichHandler

logging.basicConfig(
    level="DEBUG",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, tracebacks_suppress=[click])],
)

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def main() -> None:

    logging.info("Generating Perlin noise...")

    noise_generator = Noise(
        settings.noise.width,
        settings.noise.height,
        settings.noise.scale,
        settings.noise.seed,
    )

    optimized_noise = torch.compile(noise_generator.fractal_noise_2d)
    _ = optimized_noise(settings.noise.num_octaves)

    with Timer() as t:
        noise = optimized_noise(settings.noise.num_octaves)

    logging.info(f"Noise Generation Executed In {t.elapsed * 1000:.2f} Miliseconds!")
    tensor.to_image(
        noise,
        "Simplex Noise",
        settings.render.output_path,
        settings.render.color_map,
        settings.render.export_dpi,
        settings.render.show_plot,
    )


if __name__ == "__main__":
    main()
