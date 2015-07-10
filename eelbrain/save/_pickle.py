# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import os
import cPickle

from .._utils import ui

__all__ = ('pickle',)


def pickle(obj, dest=None, protocol=cPickle.HIGHEST_PROTOCOL):
    """Pickle a Python object.

    Parameters
    ----------
    dest : None | str
        Path to destination where to save the  file. If no destination is
        provided, a file dialog is shown. If a destination without extension is
        provided, '.pickled' is appended.
    protocol : int
        Pickle protocol (default is HIGHEST_PROTOCOL).
    """
    if dest is None:
        filetypes = [("Pickled Python Objects (*.pickled)", '*.pickled')]
        dest = ui.ask_saveas("Pickle Destination", "", filetypes)
        if dest is False:
            raise RuntimeError("User canceled")
        else:
            print 'dest=%r' % dest
    else:
        dest = os.path.expanduser(dest)
        if not os.path.splitext(dest)[1]:
            dest += '.pickled'

    with open(dest, 'wb') as fid:
        cPickle.dump(obj, fid, protocol)
