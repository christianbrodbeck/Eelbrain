# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Fix up surfer.Brain"""
from distutils.version import LooseVersion
import os
import sys

from ._base import backend

# pyface imports: set GUI backend (ETS don't support wxPython 3.0)
if backend['ets_toolkit']:
    os.environ['ETS_TOOLKIT'] = backend['ets_toolkit']

# surfer imports, lower screen logging level
first_import = 'surfer' not in sys.modules
import surfer
if first_import:
    from ..mne_fixes import reset_logger
    reset_logger(surfer.utils.logger)
from surfer import Brain as SurferBrain
from ._brain_mixin import BrainMixin


def assert_can_save_movies():
    if LooseVersion(surfer.__version__) < LooseVersion('0.6'):
        raise ImportError("Saving movies requires PySurfer 0.6")


class Brain(BrainMixin, SurferBrain):

    def __init__(self, data, *args, **kwargs):
        BrainMixin.__init__(self, data)
        SurferBrain.__init__(self, *args, **kwargs)

        from traits.trait_base import ETSConfig
        self._prevent_close = ETSConfig.toolkit == 'wx'

    def close(self):
        "Prevent close() call that causes segmentation fault"
        if not self._prevent_close:
            SurferBrain.close(self)
