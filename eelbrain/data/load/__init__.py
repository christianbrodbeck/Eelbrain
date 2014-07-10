"""
Tools for loading data.

The following submodules are available:

eyelink:
    Load eyelink ``.edf`` files to datasets. Requires eyelink api available from
    SR Research

fiff:
    Load mne fiff files to datasets and as mne objects (requires mne-python)

txt:
    Load datasets and vars from text files

"""

from . import besa
from . import eyelink
from . import fiff
from . import txt

from .txt import tsv
from ._pickle import unpickle
