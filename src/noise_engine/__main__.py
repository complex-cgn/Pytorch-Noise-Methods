import logging

import click
from rich.logging import RichHandler

import noise_engine.core.noise as noise
from noise_engine.settings import settings
from noise_engine.utils import tensor
from noise_engine.utils.timer import Timer

logging.basicConfig(
    level="DEBUG",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, tracebacks_suppress=[click])],
)

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def main() -> None:

    logging.info("Generating Perlin noise...")

    with Timer() as t:
        """output = noise.FractalNoise2D(
            scale=settings.noise.scale,
            octaves=settings.noise.num_octaves,
            shape=(settings.noise.height, settings.noise.width),
            seed=settings.noise.seed,
        )()"""
        output = noise.WhiteNoise2D(
            shape=(settings.noise.height, settings.noise.width),
        )()
        
    logging.info(f"Noise Generation Executed In {t.elapsed * 1000:.2f} Miliseconds!")
    tensor.to_image(
        output,
        "Fractal Noise",
        settings.render.output_path,
        settings.render.color_map,
        settings.render.export_dpi,
        settings.render.show_plot,
    )


if __name__ == "__main__":
    main()
