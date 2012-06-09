"""
Modules for loading data. Submodules are not preloaded and need be imported
like::

    >>> import load.

The following submodules are available:

eyelink:
    Load eyelink .edf files to datasets. Requires eyelink api available from
    SR Research 

fiff:
    Load mne fiff files to datasets and as mne objects (requires mne)  

txt:
    Load datasets and vars from text files

"""