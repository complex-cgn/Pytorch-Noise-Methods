import logging

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.warning("Device: %s", device)

if device.type == "cpu":
    logging.warning("CPU device detected")
