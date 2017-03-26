# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from distutils.version import LooseVersion
import mne


MNE_VERSION = LooseVersion(mne.__version__)
MNE_14_OR_GREATER = MNE_VERSION >= LooseVersion('0.14')
if MNE_VERSION < LooseVersion('0.13'):
    raise ImportError("Eelbrain requires mne-python version 0.23 or later, "
                      "found %s" % mne.__version__)
