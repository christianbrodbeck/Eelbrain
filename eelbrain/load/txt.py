'''
Tools for loading data from text files.

.. autosummary::
   :toctree: generated

   tsv
   var
'''
import os
import re

import numpy as np

from .._utils import ui
from .. import _data_obj as _data


__all__ = ['tsv', 'var']

float_pattern = re.compile("^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$")

def _str_is_float(x):
    if float_pattern.match(x):
        return True
    elif x.lower() == 'nan':
        return True
    else:
        return False


# could use csv module (http://docs.python.org/2/library/csv.html) but it
# currently does not support unicode
def tsv(path=None, names=True, types='auto', delimiter='\t', skiprows=0,
        start_tag=None, ignore_missing=False):
    """
    Load a :class:`Dataset` from a tab-separated values file.

    Parameters
    ----------
    path : None | str
        Path to the tsv file. If None, a system file dialog will open.
    names : list of str | bool
        * ``['name1', ...]`` use these names
        * ``True``: look for names on the first line of the file
        * ``False``: use "v1", "v2", ...
    types : 'auto' | list of int
        * ``'auto'`` -> import as Var if all values can be converted float,
          otherwise as Factor
        * list of 0=auto, 1=Factor, 2=Var. e.g. ``[0,1,1,0]``
    delimiter : None | str
        Value delimiting cells in the input file (default: ``'\\t'`` (tab);
        None = any whitespace).
    skiprows : int
        Skip so many rows at the beginning of the file (for tsv files with
        headers). Column names (if names==True) are expected to come after
        the skipped rows. Skiprows is applied after start_tag.
    start_tag : None | str
        Alternative way to skip header rows. The table is assumed to start
        on the line following the last line in the file that starts with
        ``start_tag``.
    ignore_missing : bool
        Ignore rows with missing values (default False). Append ``NaN`` for
        numerical and ``""`` for categorial variables.
    """
    if path is None:
        path = ui.ask_file("Load TSV", "Select tsv file to import as Dataset")
        if not path:
            return

    with open(path) as fid:
        lines = fid.readlines()
        if len(lines) == 1:
            # tsv file exported by excel had carriage return only
            text = lines[0]
            if text.count('\r') > 1:
                lines = text.split('\r')

    # find start position
    if start_tag:
        start = 0
        for i, line in enumerate(lines, 1):
            if line.startswith(start_tag):
                start = i
        if start:
            lines = lines[start:]
    if skiprows:
        lines = lines[skiprows:]

    # read / create names
    if names == True:
        head_line = lines.pop(0)
        names = head_line.split(delimiter)
        names = [n.strip().strip('"') for n in names]

    # separate lines into values
    rows = [[v.strip() for v in line.split(delimiter)] for line in lines]

    row_lens = set(len(row) for row in rows)
    if len(row_lens) > 1 and not ignore_missing:
        msg = ("Not all rows have same number of entries. Set ignore_missing "
               "to True in order to ignore this error.")
        raise ValueError(msg)
    n_cols = max(row_lens)

    if names:
        if len(names) != n_cols:
            msg = ("The number of names in the header (%i) does not "
                   "correspond to the number of columns in the table (%i)"
                   % (len(names), n_cols))
            raise ValueError(msg)
    else:
        names = ['v%i' % i for i in xrange(n_cols)]

    if types in ('auto', None, False, True):
        types = [0] * n_cols
    else:
        assert len(types) == n_cols

    data = np.empty((len(rows), n_cols), object)
    for r, line in enumerate(rows):
        for c, v in enumerate(line):
            for str_del in ["'", '"']:
                if len(v) > 0 and v[0] == str_del:
                    v = v.strip(str_del)
                    types[c] = 1
            data[r, c] = v

    ds = _data.Dataset(name=os.path.basename(path))

    # convert values to data-objects
    np_vars = vars(np)
    bool_dict = {'True': True, 'False': False, None: False}
    for name, values, type_ in zip(names, data.T, types):
        if type_ == 1:
            dob = _data.Factor(values, labels={None: ''}, name=name)
        elif all(v in ('True', 'False', None) for v in values):
            values = [bool_dict[v] for v in values]
            dob = _data.Var(values, name=name)
        elif all(v is None or _str_is_float(v) for v in values):
            values = [np.nan if v is None else eval(v, np_vars) for v in values]
            dob = _data.Var(values, name=name)
        elif type_ == 2:
            err = ("Could not convert all values to float: %s" % values)
            raise ValueError(err)
        else:
            dob = _data.Factor(values, labels={None: ''}, name=name)
        ds.add(dob)
    return ds


def var(path=None, name=None):
    """
    Load a :class:`Var` object from a text file by splitting at white-spaces.

    Parameters
    ----------
    path : str(path) | None
        Source file. If None, a system file dialog is opened.
    name : str | None
        Name for the Var.
    """
    if path is None:
        path = ui.ask_file("Select Var File", "()")

    FILE = open(path)
    lines = FILE.read().split()
    FILE.close()
    is_bool = all(line in ['True', 'False'] for line in lines)

    if is_bool:
        x = np.genfromtxt(path, dtype=bool)
    else:
        x = np.loadtxt(path)

    return _data.Var(x, name=None)
