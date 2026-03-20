import logging

import click
import noise_engine.core.noise as noise
from noise_engine.settings import get_settings
from noise_engine.utils.dynamic_3d_plotter import Dynamic3DPlotter
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
    """Main entry point for noise generation."""
    settings = get_settings()

    logging.info("Generating Perlin noise...")

    with Timer() as t:
        output = noise.FractalNoise3D(
            scale=settings.noise.scale,
            octaves=settings.noise.num_octaves,
            shape=(20, 20, 20),
            seed=settings.noise.seed,
        )()

    logging.info(f"Noise Generation Executed In {t.elapsed * 1000:.2f} Miliseconds!")
    Dynamic3DPlotter().plot(
        output, mode="scatter", title="Fractal Noise 3D Scatter Plot"
    )


if __name__ == "__main__":
    main()
