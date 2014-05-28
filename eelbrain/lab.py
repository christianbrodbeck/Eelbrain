'''
The eelbrain.lab module contains a collection of Eelbrain components that
are useful for interactive data analysis.
'''
from .data import *
from .fmtxt import Report

# mne
import eelbrain.data.load.fiff

# mayavi
try:
    import eelbrain.data.plot.brain
except:
    globals().setdefault('err', []).append('plot.brain (mayavi)')

from . import ui
from . import gui
