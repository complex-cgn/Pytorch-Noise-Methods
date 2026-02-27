import logging

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.info("Device: %s", device)

if device.type == "cpu":
    logging.warning("CPU device detected. Performance may be significantly slower than GPU.")
