'''
Created on Jun 9, 2012

@author: christian
'''
import os
import re

import numpy as np

from ... import ui
from .. import data_obj as _data


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
        start_tag=None):
    """
    Load a :class:`Dataset` from a tab-separated values file.

    Parameters
    ----------
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
        the skipped rows.
    start_tag : None | str
        Alternative way to skip header rows. The table is assumed to start
        on the line following the last line in the file that starts with
        ``start_tag``.
    """
    if path is None:
        path = ui.ask_file("Load TSV", "Select tsv file to import as Dataset")
        if not path:
            return

    with open(path) as fid:
        lines = fid.readlines()

    # find start position
    if skiprows:
        lines = lines[skiprows:]
    if start_tag:
        start = 0
        for i, line in enumerate(lines, 1):
            if line.startswith(start_tag):
                start = i
        if start:
            lines = lines[start:]

    # read / create names
    if names == True:
        head_line = lines.pop(0)
        names = head_line.split(delimiter)
        names = [n.strip().strip('"') for n in names]

    # read table body
    rows = []
    for line in lines:
        values = []
        for v in line.split(delimiter):
            v = v.strip()
            values.append(v)
        rows.append(values)

    n_vars = len(rows[0])

    if not names:
        names = ['v%i' % i for i in xrange(n_vars)]

    n = len(names)
    # decide whether to drop first column
    if n_vars == n:
        start = 0
    elif n_vars == n + 1:
        start = 1
    else:
        raise ValueError("number of header different from number of data")

    if types in ['auto', None, False, True]:
        types = [0] * n
    else:
        assert len(types) == n

    # prepare for reading data
    data = []
    for _ in xrange(n):
        data.append([])

    # read rest of the data
    for line in rows:
        for i, v in enumerate(line[start:]):
            for str_del in ["'", '"']:
                if len(v) > 0 and v[0] == str_del:
                    v = v.strip(str_del)
                    types[i] = 1
            data[i].append(v)

    ds = _data.Dataset(name=os.path.basename(path))

    # convert values to data-objects
    np_vars = vars(np)
    for name, values, type_ in zip(names, data, types):
        if type_ == 1:
            dob = _data.Factor(values, name=name)
        elif all(v in ('True', 'False') for v in values):
            values = [{'True': True, 'False': False}[v] for v in values]
            dob = _data.Var(values, name=name)
        elif all(_str_is_float(v) for v in values):
            values = [eval(v, np_vars) for v in values]
            dob = _data.Var(values, name=name)
        elif type_ == 2:
            err = ("Could not convert all values to float: %s" % values)
            raise ValueError(err)
        else:
            dob = _data.Factor(values, name=name)
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
