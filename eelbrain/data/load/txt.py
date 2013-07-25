'''
Created on Jun 9, 2012

@author: christian
'''
import os

import numpy as np

from ... import ui
from .. import data_obj as _data


__all__ = ['tsv', 'var']


def tsv(path=None, names=True, types='auto', empty='nan', delimiter=None,
        skiprows=0, start_tag=None):
    """
    Load a :class:`dataset` from a tab-separated values file.

    Parameters
    ----------
    names : list of str | bool
        * ``['name1', ...]`` use these names
        * ``True``: look for names on the first line of the file
        * ``False``: use "v1", "v2", ...
    types : 'auto' | list of int
        * ``'auto'`` -> import as var if all values can be converted float,
          otherwise as factor
        * list of 0=auto, 1=factor, 2=var. e.g. ``[0,1,1,0]``
    empty :
        value to substitute for empty cells
    delimiter : str
        value delimiting cells in the input file (None = any whitespace;
        e.g., ``'\\t'``)
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
        path = ui.ask_file("Load TSV", "Select tsv file to import as dataset")
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
            if not v:
                v = empty
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
                if v[0] == str_del:
                    v = v.strip(str_del)
                    types[i] = 1
            data[i].append(v)

    ds = _data.dataset(name=os.path.basename(path))

    for name, values, force_type in zip(names, data, types):
        v = np.array(values)
        if force_type in [0, 2]:
            try:
                v = v.astype(float)
                fid = _data.var(v, name=name)
            except:
                fid = _data.factor(v, name=name)
        else:
            fid = _data.factor(v, name=name)
        ds.add(fid)

    return ds


def var(path=None, name=None):
    """
    Loads a ``var`` object from a text file by splitting at white-spaces.

    path : str(path) | None
        Source file. If None, a system file dialog is opened.
    name : str | None
        Name for the var.

    """
    if path is None:
        path = ui.ask_file("Select var File", "()")

    FILE = open(path)
    lines = FILE.read().split()
    FILE.close()
    is_bool = all(line in ['True', 'False'] for line in lines)

    if is_bool:
        x = np.genfromtxt(path, dtype=bool)
    else:
        x = np.loadtxt(path)

    return _data.var(x, name=None)
