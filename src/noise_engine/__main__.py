import logging

import click
import noise_engine.core.noise as noise
from noise_engine.settings import get_settings
from noise_engine.core.utils.dynamic_3d_plotter import Dynamic3DPlotter
from noise_engine.core.utils.timer import Timer
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

    logging.info("Generating Fractal noise...")

    with Timer() as t:
        output = noise.FractalNoise2D(
            scale=settings.noise.scale,
            octaves=settings.noise.num_octaves,
            shape=(100, 100),
            seed=settings.noise.seed,
        )()

    logging.info(f"Noise Generation Executed In {t.elapsed * 1000:.2f} Miliseconds!")
    Dynamic3DPlotter().plot(
        output, mode="scatter", title="Fractal Noise 3D Scatter Plot"
    )


if __name__ == "__main__":
    main()
