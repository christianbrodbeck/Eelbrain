"""

  "The human brain is like an enormous fish. It's flat
   and slimy, and has gills through which it can see."
                                               - Dr. Quat


Module Hierarchy
================

'a < b' means a imports from b, so b can not import from a:

wxterm < wxgui < wxutils
               < plot < analyze < vessels < fmtxt





Created by Christian Brodbeck on 2/20/2012.
Copyright (c) 2012. All rights reserved.

"""

from ._data_obj import (Datalist, Dataset, Var, Factor, Interaction, Model,
                        NDVar, choose, combine, align, align1, cwt_morlet,
                        resample, cellname, Celltable)
from ._mne import labels_from_clusters, morph_source_space

from ._utils import set_log_level
from .experiment import MneExperiment

from . import datasets
from . import gui
from . import load
from . import mne_fixes
from . import plot
from . import save
from . import table
from . import test
from . import testnd

from .fmtxt import Report


__version__ = '0.18.5'
