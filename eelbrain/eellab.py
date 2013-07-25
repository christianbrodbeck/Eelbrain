'''
The eelbrain.eellab module contains a collection of Eelbrain components that
are useful for
interactive data analysis, so for interactive usage you could::

    >>> from eelbrain.eellab import *
    # or
    >>> import eelbrain.eellab as EL


**classes and constructors:**

factor
    standard class for categorial data
var
    standard class for univariate scalar data

...


**Modules:**

load
    functions for loading various types of data
process
    functions for processing ndvar objects (such as baseline correction, pca)

...




Created on Mar 27, 2012

@author: christian
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
