import logging

import torch

from .settings import settings
from .core.noise import Noise
from .utils import tensor
from .utils.timer import Timer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Generating Perlin noise...")

noise_generator = Noise(
    settings.noise_options.width,   
    settings.noise_options.height,
    settings.noise_options.scale,
    settings.noise_options.seed,
)

optimized_noise = torch.compile(noise_generator.fractal_noise_2d)
_ = optimized_noise(settings.noise_options.num_octaves)

with Timer() as t:
    xs, ys = torch.meshgrid(
        torch.linspace(
            0, 12, settings.noise_options.width, device=noise_generator.device
        ),
        torch.linspace(
            0, 12, settings.noise_options.height, device=noise_generator.device
        ),
        indexing="ij",
    )
    perm = torch.randperm(256, device=noise_generator.device)
    perm = torch.cat([perm, perm])

    noise = noise_generator.simplex_noise_2d(xs, ys, perm)

    # noise = optimized_noise(config.OCTAVES)

logger.info(f"Noise Generation Executed In {t.elapsed * 1000:.2f} Miliseconds!")
tensor.to_image(
    noise,
    "Simplex Noise",
    settings.render_options.output_path,
    settings.render_options.color_map,
    settings.render_options.export_dpi,
    settings.render_options.show_plot,
)
