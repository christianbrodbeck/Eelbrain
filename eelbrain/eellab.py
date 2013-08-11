'''
The eelbrain.eellab module contains a collection of Eelbrain components that
are useful for interactive data analysis.
'''
from .data import *

# mne
import eelbrain.data.load.fiff
from .utils.mne_utils import split_label

# mayavi
try:
    import eelbrain.data.plot.brain
    import eelbrain.data.plot.coreg
except:
    globals().setdefault('err', []).append('plot.brain (mayavi)')

from . import ui
from . import gui
