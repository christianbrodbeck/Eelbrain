# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import os

from .._utils import ui

__all__ = ['txt']


def txt(iterator, fmt='%s', delim=os.linesep, dest=None):
    """
    Writes any object that supports iteration to a text file.

    Parameters
    ----------
    iterator : iterator
        Object that iterates over values to be saved
    fmt : fmt-str
        format-string which is used to format the iterator's values
    delim : str
        the delimiter which is inserted between values
    dest : str(path) | None
        The destination; if None, a system save-as dialog is displayed
    """
    if dest is None:
        name = repr(iterator)[:20]
        msg = "Save %s..." % name
        dest = ui.ask_saveas(msg, msg, [("Plain Text File", '*.txt')])

    if dest:
        with open(dest, 'w') as FILE:
            FILE.write(delim.join(fmt % v for v in iterator))
