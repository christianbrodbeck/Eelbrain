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
var_from_dict
    construct a var from a dictionary

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
from .vessels.data import (factor,
                           cellname,
                           var,
                           var_from_dict,
                           var_from_apply,
                           ndvar,
                           resample,
                           dataset,
                           combine,
                           align,
                           interaction,
                           )

from .vessels.structure import celltable

from .vessels import process, datasets

from .analyze import (test,
                      testnd,
                      table)

import ui
import plot
try:  # mayavi
    import plot.brain
    import plot.coreg
except:
    globals().setdefault('err', []).append('plot.brain (mayavi)')

import load.txt
import load.eyelink
try:
    import load.fiff
except:
    globals().setdefault('err', []).append('load.fiff (mne)')
import load.kit

import save

from .utils import statfuncs
