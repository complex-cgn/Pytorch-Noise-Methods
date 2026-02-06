import logging
import os

import matplotlib.pyplot as plt
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def tensor_to_image(
    image_tensor: torch.Tensor, output_path: str, color_map: str, dpi: int
):
    """Tensor to Image converter with matplotlib"""

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        logger.info("Output folder can't found creating new one...")
        os.makedirs(output_dir)

    # Save as image
    plt.figure(figsize=(2, 2))
    plt.imshow(image_tensor.cpu().numpy(), cmap="gray", origin="upper")
    plt.axis("off")
    plt.title("Perlin Noise")
    plt.savefig(output_path, dpi=dpi)
    plt.close()

    logger.info(f"Noise generated and saved to {output_path}")
