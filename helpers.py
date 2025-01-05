import logging
from logging import Logger
from sys import platform

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