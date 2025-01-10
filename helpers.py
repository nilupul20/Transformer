import logging
from logging import Logger
from pathlib import Path
import os

from transformer_config import TransformerConfig

def get_logger(model_dir: str, log_file: str = "train.log") -> Logger:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.setLevel(level=logging.DEBUG)
        fh = logging.FileHandler("{}/{}".format(model_dir, log_file))
        fh.setLevel(level=logging.DEBUG)
        logger.addHandler(fh)
        formatter = logging.Formatter("%(asctime)s %(message)s")
        fh.setFormatter(formatter)

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)
        logging.getLogger("").addHandler(sh)
        return logger

 
def get_checkpoint_path(cfg: TransformerConfig, epoch: str):
    model_folder = f"{cfg.model_folder}"
    model_filename = f"{cfg.model_basename}{epoch}.pt"
    return str(Path('.') / 'runs' / model_folder / model_filename)


def get_console_width():
    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80
    return console_width