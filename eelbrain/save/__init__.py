"""
Helper functions for saving data in various formats.

"""

import os
import cPickle

from eelbrain import ui
from _besa import *
from _txt import *


__hide__ = ['ui', 'cPickle', 'os']



def pickle(obj, dest=None, protocol=cPickle.HIGHEST_PROTOCOL):
    """
    Pickle an object.

    dest : None | str
        Path to destination where to save the  file. If no destination is
        provided, a file dialog opens. If a destination without extension is
        provided, '.pickled' is appended.
    protocol : int
        Pickle protocol (default is HIGHEST_PROTOCOL).

    """
    if dest is None:
        ext = [('pickled', "Pickled Python Object")]
        dest = ui.ask_saveas("Pickle Destination", "", ext=ext)
        if dest:
            print 'dest=%r' % dest
        else:
            return
    else:
        dest = os.path.expanduser(dest)
        if not os.path.splitext(dest)[1]:
            dest += '.pickled'

    with open(dest, 'w') as FILE:
        cPickle.dump(obj, FILE, protocol)
