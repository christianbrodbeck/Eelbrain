"""Helper functions for dealing with logging

By default, mne's logger has a highly verbose stream handler logging to the
screen. Here this screen handler is set to a higher level (WARNING), but only
if Eelbrain is the primary import (i.e., it is imported before mne) and if the
user has not stored another configuration.
"""
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from logging import WARNING, FileHandler, Formatter, StreamHandler, getLogger
import sys
first_mne_import = 'mne' not in sys.modules

import mne

from .._utils import log_level


if first_mne_import:
    # lower mne's screen logging level, but defer to user-defined setting
    level = mne.get_config('MNE_LOGGING_LEVEL', WARNING)
    formatter = Formatter("%(levelname)-8s %(name)s:%(message)s")
    for h in mne.utils.logger.handlers:
        if isinstance(h, StreamHandler):
            h.setFormatter(formatter)
            if h.level < level:
                h.setLevel(level)


