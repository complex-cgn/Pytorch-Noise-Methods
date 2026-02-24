import logging
import os

import matplotlib.pyplot as plt
import torch


def to_image(
    image_tensor: torch.Tensor,
    plot_title: str,
    output_path: str,
    color_map: str,
    dpi: int,
    show_plot: bool,
):
    """
    Tensor-to-image conversion utility.

    Supports saving tensors as images using matplotlib, with optional
    color mapping and plot display.
    """

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        logging.info("Output folder can't found creating new one...")
        os.makedirs(output_dir)

    # Save as image
    plt.figure(figsize=(2, 2))
    plt.imshow(image_tensor.cpu().numpy(), cmap=color_map, origin="upper")
    plt.axis("off")
    plt.title(plot_title, fontname="DejaVu Sans", fontsize=16)
    plt.savefig(output_path, dpi=dpi)

    if show_plot:
        plt.show()
    plt.close()

    logging.info(f"Noise generated and saved to {output_path}")
