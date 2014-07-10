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

from ._utils import _set_log_level

from .data import *
from .fmtxt import Report

from . import gui


__version__ = '0.4dev'
