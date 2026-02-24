import logging

import click
import torch
from rich.logging import RichHandler

from .core.noise import Noise
from .settings import settings
from .utils import tensor
from .utils.timer import Timer


def main() -> None:

    logging.basicConfig(
        level="DEBUG",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, tracebacks_suppress=[click])],
    )

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.info("Generating Perlin noise...")

    noise_generator = Noise(
        settings.noise_options.width,
        settings.noise_options.height,
        settings.noise_options.scale,
        settings.noise_options.seed,
    )

    optimized_noise = torch.compile(noise_generator.fractal_noise_2d)
    _ = optimized_noise(settings.noise_options.num_octaves)

    with Timer() as t:
        noise = optimized_noise(settings.noise_options.num_octaves)

    logging.info(f"Noise Generation Executed In {t.elapsed * 1000:.2f} Miliseconds!")
    tensor.to_image(
        noise,
        "Simplex Noise",
        settings.render_options.output_path,
        settings.render_options.color_map,
        settings.render_options.export_dpi,
        settings.render_options.show_plot,
    )


if __name__ == "__main__":
    main()
