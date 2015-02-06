'''
Create tables for data objects.

'''
from __future__ import division

from itertools import izip
import re

import numpy as np

from . import fmtxt
from ._data_obj import (ascategorial, asvar, assub, isfactor, isinteraction,
                        Dataset, Factor, Var, Celltable, cellname, combine)


def difference(Y, X, c1, c0, match, by=None, sub=None, ds=None):
    """Subtract data in one cell from another

    Parameters
    ----------
    Y : Var | NDVar
        Dependent variable.
    X : categorial
        Model for subtraction.
    c1 : str | tuple
        Name of the cell in X that forms the minuend.
    c0 : str | tuple
        Name of the cell in X that is to be subtracted from c1.
    match : categorial
        Units over which measurements were repeated.
    by : None | categorial
        Grouping variable to define cells for which to calculate differences.
    sub : None | index
        Only include a subset of the data.
    ds : None | Dataset
        If a Dataset is specified other arguments can be str instead of
        data-objects and will be retrieved from ds.

    Returns
    -------
    diff : Dataset
        Dataset with the difference between c1 and c0 on Y.
    """
    sub = assub(sub, ds)
    X = ascategorial(X, sub, ds)
    out = Dataset()
    if by is None:
        ct = Celltable(Y, X, match, sub, ds=ds)
        out.add(ct.groups[c1])
        if not ct.all_within:
            raise ValueError("Design is not fully balanced")
        out[ct.Y.name] = ct.data[c1] - ct.data[c0]
    else:
        by = ascategorial(by, sub, ds)
        ct = Celltable(Y, X % by, match, sub, ds=ds)
        if not ct.all_within:
            raise ValueError("Design is not fully balanced")

        dss = []
        if isinstance(c1, str):
            c1 = (c1,)
        if isinstance(c0, str):
            c0 = (c0,)
        for cell in by.cells:
            if isinstance(cell, str):
                cell = (cell,)
            cell_ds = Dataset()
            cell_ds.add(ct.groups[c1 + cell])
            cell_ds[ct.Y.name] = ct.data[c1 + cell] - ct.data[c0 + cell]
            if isfactor(by):
                cell_ds[by.name, :] = cell[0]
            else:
                for b, c in zip(by.base, cell):
                    cell_ds[b.name, :] = c
            dss.append(cell_ds)
        out = combine(dss)

    return out


def frequencies(Y, X=None, of=None, sub=None, ds=None):
    """Calculate frequency of occurrence of the categories in Y

    Parameters
    ----------
    Y : categorial
        Factor with values whose frequencies are of interest.
    X : None | categorial
        Optional model defining cells for which frequencies are displayed
        separately.
    of : None | categorial
        With `X` constant within `of`, only count frequencies for each value
        in `of` once. (Compress Y and X before calculating frequencies.)
    sub : None | index
        Only use a subset of the data.
    ds : Dataset
        If ds is specified, other parameters can be strings naming for
        variables in ds.

    Returns
    -------
    freq : Dataset
        Dataset with frequencies.
    """
    sub = assub(sub, ds)
    Y = ascategorial(Y, sub, ds)
    if X is not None:
        X = ascategorial(X, sub, ds)
    if of is not None:
        of = ascategorial(of, sub, ds)
        Y = Y.aggregate(of)
        if X is not None:
            X = X.aggregate(of)

    # find name
    if getattr(Y, 'name', None):
        name = "Frequencies of %s" % Y.name
    else:
        name = "Frequencies"

    # special case
    if X is None:
        out = Dataset(name=name)
        out['cell'] = Factor(Y.cells, random=Y.random)
        n = np.fromiter((np.sum(Y == cell) for cell in Y.cells), int,
                        len(Y.cells))
        out['n'] = Var(n)
        return out

    # header
    if getattr(X, 'name', None):
        name += ' by %s' % X.name
    out = Dataset(name=name)

    if isinteraction(X):
        for i, f in enumerate(X.base):
            random = getattr(f, 'random', False)
            out[f.name] = Factor((c[i] for c in X.cells), random=random)
    else:
        out[X.name] = Factor(X.cells)

    y_idx = {cell: Y == cell for cell in Y.cells}
    x_idx = {cell: X == cell for cell in X.cells}
    for y_cell in Y.cells:
        n = (np.sum(np.logical_and(y_idx[y_cell], x_idx[x_cell]))
             for x_cell in X.cells)
        name = cellname(y_cell, '_')
        out[name] = Var(np.fromiter(n, int, len(X.cells)))

    return out


def melt(name, cells, cell_var_name, ds):
    """
    Restructure a Dataset such that a measured variable is in a single row

    Restructure a Dataset with a certain variable represented in several
    columns into a longer dataset in which the variable is represented in a
    single column along with an identifying variable.

    Parameters
    ----------
    name : str
        Name of the variable in the new Dataset.
    cells : sequence of str | str
        Names of the columns representing the variable in the input Dataset.
        Names can either pe specified explicitly as a sequence of str, or
        implicitly as a str containing '%i' for an integer.
    cell_var_name : str
        Name of the variable to contain the cell identifier.
    ds : Dataset
        Input Dataset.
    """
    # find source cells
    if isinstance(cells, basestring):
        cell_expression = cells
        cells = []
        cell_values = []
        if '%i' in cell_expression:
            pattern = cell_expression.replace('%i', '(\d+)')
            for key in ds:
                m = re.match(pattern, key)
                if m:
                    cells.append(key)
                    cell_values.append(int(m.group(1)))
    else:
        cell_values = cells

    # melt the Dataset
    dss = []
    for cell, cell_value in izip(cells, cell_values):
        cell_ds = ds.copy()
        cell_ds.rename(cell, name)
        for src in cells:
            if src != cell:
                del cell_ds[src]
        cell_ds[cell_var_name, :] = cell_value
        dss.append(cell_ds)
    out = combine(dss)
    return out



def stats(Y, row, col=None, match=None, sub=None, fmt='%.4g', funcs=[np.mean],
          ds=None):
    """
    Make a table with statistics.

    Parameters
    ----------
    Y : Var
        Dependent variable.
    row : categorial
        Model specifying rows
    col : categorial | None
        Model specifying columns.
    funcs : list of callables
        A list of statistics functions to show (all functions must take an
        array argument and return a scalar).
    ds : Dataset
        If a Dataset is provided, Y, row, and col can be strings specifying
        members.


    Examples
    --------
    >>> ds = datasets.get_uts()
    >>> table.stats('Y', 'A', 'B', ds=ds)
                B
         -----------------
         b0        b1
    ----------------------
    a0   0.1668    -0.3646
    a1   -0.4897   0.8746

    >>> A.table.stats(Y, condition, funcs=[np.mean, np.std])
    Condition   mean     std
    ----------------------------
    control     0.0512   0.08075
    test        0.2253   0.2844

    """
    Y = asvar(Y, ds=ds)
    row = ascategorial(row, ds=ds)

    if col is None:
        ct = Celltable(Y, row, sub=sub, match=match)

        # table header
        n_disp = len(funcs)
        table = fmtxt.Table('l' * (n_disp + 1))
        table.cell('Condition', 'bf')
        for func in funcs:
            table.cell(func.__name__, 'bf')
        table.midrule()

        # table entries
        for cell in ct.cells:
            data = ct.data[cell]
            table.cell(cell)
            for func in funcs:
                table.cell(fmt % func(data.x))
    else:
        col = ascategorial(col, ds=ds)
        ct = Celltable(Y, row % col, sub=sub, match=match)

        N = len(col.cells)
        table = fmtxt.Table('l' * (N + 1))

        # table header
        table.cell()
        table.cell(col.name, width=N, just='c')
        table.midrule(span=(2, N + 1))
        table.cell()

        table.cells(*col.cells)
        table.midrule()

        # table body
        fmt_n = fmt.count('%')
        if fmt_n == 1:
            fmt_once = False
        elif len(funcs) == fmt_n:
            fmt_once = True
        else:
            raise ValueError("fmt does not match funcs")

        for Ycell in row.cells:
            table.cell(Ycell)
            for Xcell in col.cells:
                # construct address
                a = ()
                if isinstance(Ycell, tuple):
                    a += Ycell
                else:
                    a += (Ycell,)
                if isinstance(Xcell, tuple):
                    a += Xcell
                else:
                    a += (Xcell,)

                # cell
                data = ct.data[a]
                values = (f(data.x) for f in funcs)
                if fmt_once:
                    txt = fmt % values
                else:
                    txt = ', '.join((fmt % v for v in values))

                table.cell(txt)

    return table



def repmeas(Y, X, match, sub=None, ds=None):
    """
    Create a repeated-measures table

    Parameters
    ----------
    Y :
        Dependent variable (can be model with several dependents).
    X : categorial
        Model defining the cells that should be restructured into variables.
    match : categorial
        Model identifying the source of the measurement across repetitions,
        i.e. the model that should be retained.
    sub :
        boolean array specifying which values to include (generate e.g.
        with 'sub=T==[1,2]')
    ds : None | Dataset
        If a Dataset is specified other arguments can be str instead of
        data-objects and will be retrieved from ds.

    Returns
    -------
    rm_table : Dataset
        Repeated measures table.
    """
    ct = Celltable(Y, X, match, sub, ds=ds)
    if not ct.all_within:
        raise ValueError("Incomplete data")

    out = Dataset()
    x = ct.groups.values()[0]
    if isinteraction(x):
        for f in x.base:
            out.add(f)
    else:
        out[ct.match.name] = x

    for cell in ct.X.cells:
        out[cellname(cell, '_')] = ct.data[cell]

    return out
