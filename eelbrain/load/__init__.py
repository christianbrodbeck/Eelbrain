"""
Modules for loading data.
The following submodules are available:

brainvision:
    Load brainvision ``.vhdr`` header files (no other files supported yet)

eyelink:
    Load eyelink ``.edf`` files to datasets. Requires eyelink api available from
    SR Research

fiff:
    Load mne fiff files to datasets and as mne objects (requires mne-python)

txt:
    Load datasets and vars from text files

"""

import cPickle as _pickle

from .. import ui as _ui


def unpickle(fname=None):
    if fname is None:
        ext = [('pickled', "Pickles"), ('*', "all files")]
        fname = _ui.ask_file("Select File to Unpickle", "Select a pickled "
                             "file to unpickle", ext=ext)
    if fname is False:
        raise IOError("User canceled")

    return _pickle.load(open(fname))
