# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from logging import NOTSET
import sys
first_mne_import = 'mne' not in sys.modules


def reset_logger(logger):
    "reset mne's logging behavior to Python default"
    for handler in logger.handlers:
        logger.removeHandler(handler)
    logger.propagate = True
    logger.setLevel(NOTSET)


if first_mne_import:
    from logging import basicConfig
    basicConfig()

    from mne.utils import logger
    reset_logger(logger)
