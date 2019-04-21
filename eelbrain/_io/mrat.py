import os
import re
import glob
import itertools
from mne import read_source_estimate

from .._data_obj import Dataset, Factor
from .fiff import stc_ndvar


class DatasetSTCLoader(object):
    """
    Load source estimates on disk into Dataset for use in statistical tests

    Parameters
    ----------
    data_dir : str
        Path to directory containing stc files

    Attributes
    ----------
    data_dir : str
        Path to data directory
    subjects : tuple of str
        Subject IDs extracted from stc filenames
    factors : tuple of str
        Names of experimental factors
    levels : tuple of tuple of str
        Names of levels of each factor in `factors`
    """
    def __init__(self, data_dir, **stc_kwargs):
        if not os.path.exists(data_dir):
            raise ValueError("Directory '%s' not found." % data_dir)
        self.data_dir = data_dir
        self._stc_kwargs = stc_kwargs
        self.subjects = None
        self.levels = None
        self.factors = None
        self._n_factors = None
        self._level_lens = None
        self._find_subjects()
        self._find_level_names()

    def _all_stc_filenames(self):
        return glob.glob(os.path.join(self.data_dir, "*", "*.stc"))

    def _find_subjects(self):
        pattern = re.compile(r"[AR]\d{4}")
        stcs = self._all_stc_filenames()
        subjects = set(pattern.search(s).group() for s in stcs)
        self.subjects = tuple(subjects)

    def _find_level_names(self):
        stcs = self._all_stc_filenames()
        # condition names should be lowest level folder
        cond_dirs = list(set(s.split(os.sep)[-2] for s in stcs))
        # set number of factors based on first full condition name
        self._n_factors = len(cond_dirs[0].split("_"))
        splits = (c.split("_") for c in cond_dirs)
        # transpose to group level names by factor; keep unique
        cond_sets = list(map(set, zip(*splits)))
        self.levels = tuple(tuple(c) for c in cond_sets)  # list of tuples, not sets
        self._level_lens = [len(lev) for lev in self.levels]

    def set_factor_names(self, factors):
        """
        Set names of experimental factors

        Parameters
        ----------
        factors : list of str | tuple of str
            Factor names. Length must match the number of factors detected
            from stc filenames.
        """
        if not self.levels:
            raise RuntimeError("No level names were detected from "
                               "the files in the data directory.")
        if len(factors) != self._n_factors:
            msg = ("There were %d factors detected, but %d factor "
                   "names provided." % (self._n_factors, len(factors)))
            raise ValueError(msg)
        self.factors = tuple(factors)

    @property
    def design_shape(self):
        """Shape of experiment design, e.g. '2 x 3'"""
        if self.levels is None or self.factors is None:
            return None
        return " x ".join(map(str, self._level_lens))

    def make_dataset(self, load_stcs=True):
        """
        Create a Dataset with one case per condition per subject, and source
        estimates loaded as an NDVar.

        Any `stc_kwargs` from __init__ are passed to `eelbrain.load.fiff.stc_ndvar()`.

        Parameters
        ----------
        load_stcs : bool
            Whether to include stc data in dataset. Only False when testing
            on unreadable stc files.

        Returns
        -------
        ds : eelbrain.Dataset
            Dataset with columns 'subject' (random factor), 'src' (NDVar of stc data),
            and one Factor for each item in `self.factors`.
        """
        rows = itertools.product(self.subjects, *self.levels)
        columns = map(Factor, zip(*rows))
        col_names = ["subject"] + list(self.factors)
        ds = Dataset(zip(col_names, columns))
        ds["subject"].random = True
        stc_fnames = []
        for c in ds.itercases():
            folder = "_".join(c[i] for i in self.factors)
            exp = "{}/{}/{}*-lh.stc".format(
                self.data_dir, folder, c["subject"])
            fnames = glob.glob(exp)
            assert len(fnames) == 1
            stc_fnames.append(fnames[0])
        if load_stcs:
            stcs = list(map(read_source_estimate, stc_fnames))
            ds["src"] = stc_ndvar(stcs, **self._stc_kwargs)
        return ds
