'''
Created on Jun 9, 2012

@author: christian
'''
import os

import numpy as np

from eelbrain import ui
from eelbrain.vessels import data as _data


__all__ = ['tsv', 'var']



def tsv(path=None, names=True, types='auto', empty='nan', delimiter=None,
        skiprows=0):
    """
    returns a ``dataset`` with data from a tab-separated values file. 
    
        
    names :
    
    * ``True``: look for names on the first line of the file
    * ``['name1', ...]`` use these names
    * ``False``: use "v1", "v2", ...
        
    types :
    
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

    """
    if path is None:
        path = ui.ask_file("Select file to import as dataframe",
                           "Select file to import as dataframe")
        if not path:
            return

    with open(path) as f:
        for i in xrange(skiprows):
            f.readline()

        # read / create names
        if names == True:
            names = f.readline().split(delimiter)
            names = [n.strip().strip('"') for n in names]

        lines = []
        for line in f:
            values = []
            for v in line.split(delimiter):
                v = v.strip()
                if not v:
                    v = empty
                values.append(v)
            lines.append(values)

    n_vars = len(lines[0])

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
    for line in lines:
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
                f = _data.var(v, name=name)
            except:
                f = _data.factor(v, name=name)
        else:
            f = _data.factor(v, name=name)
        ds.add(f)

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
