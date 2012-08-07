"""
Helper functions for saving data in various formats.

"""

import os
import cPickle

from eelbrain import ui
from _besa import *
from _txt import *


__hide__ = ['ui', 'cPickle', 'os']



def pickle(obj, dest=None):
    if dest is None:
        dest = ui.ask_saveas("Pickle Destination", '', 
                             ext=[('pickled', "Pickled Python Object")])
        if dest:
            print 'dest=%r' % dest
        else:
            return
    else:
        dest = os.path.expanduser(dest)
    
    with open(dest, 'w') as FILE:
        cPickle.dump(obj, FILE)
