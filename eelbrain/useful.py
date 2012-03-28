'''
This module simply imports entities that are used often, so for interactive 
usage you could::

    >>> import eelbrain.vessels.useful as V

or::

    >>> from eelbrain.vessels.useful import *




Created on Mar 27, 2012

@author: christian
'''

from vessels.data import (factor, 
                          factor_from_dict,
                          
                          var,
                          var_from_dict,
                          var_from_apply,
                          
                          dataset,
                          )

from vessels import load
from vessels import process as proc

from analyze.glm import anova, ancova
from analyze import (test,
                     plot,
                     table) 
