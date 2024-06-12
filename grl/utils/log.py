import logging
import os

from rich.logging import RichHandler

import wandb

# Silence wandb by using the following line
# os.environ["WANDB_SILENT"] = "True"
# wandb_logger = logging.getLogger("wandb")
# wandb_logger.setLevel(logging.ERROR)

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("rich")
