"""
Tools for loading data.

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

from ... import ui as _ui
from . import besa
from . import brainvision
from . import eyelink
from . import txt
from .txt import tsv


def unpickle(file_path=None):
    """Load pickled Python objects from a file.

    Simply uses cPickle.load(), but allows using a system file dialog to select
    a file.

    Parameters
    ----------
    file_path : None | str
        Path to a pickled file. If None, a system file dialog will be used. If
        the user cancels the file dialog, a RuntimeError is raised.
    """
    if file_path is None:
        filetypes = [("Pickles (*.pickled)", '*.pickled'), ("All files", '*')]
        file_path = _ui.ask_file("Select File to Unpickle", "Select a pickled "
                                 "file to unpickle", filetypes)
    if file_path is False:
        raise RuntimeError("User canceled")

    return _pickle.load(open(file_path))
