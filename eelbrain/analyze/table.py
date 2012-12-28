'''
Creating tables for data objects.

'''
from __future__ import division

import numpy as np

from eelbrain import fmtxt

import eelbrain.vessels.data as _data
import eelbrain.vessels.structure as _structure

__hide__ = ['division', 'fmtxt', 'scipy',
            'asmodel', 'isfactor', 'asfactor', 'isvar', 'celltable',
            ]



def frequencies(Y, X=None, of=None, sub=None, title="{Yname} Frequencies", ds=None):
    """
    Display frequency of occurrence of all categories in Y in the cells
    defined by X.

    Y : categorial
        Factor with values whose frequencies are of interest.
    X : None | categorial
        Model defining cells for which frequencies are displayed.
    of : categorial
        If `X` is constant within `of`, only count frequencies for each value
        in `of` once. (Compress Y and X before calculating frequencies.)

    """
    Y = _data.ascategorial(Y, sub, ds)
    if X is not None:
        X = _data.ascategorial(X, sub, ds)
    if of is not None:
        of = _data.ascategorial(of, sub, ds)
        Y = Y.compress(of)
        if X is not None:
            X = X.compress(of)

    if X is None:
        table = fmtxt.Table('ll')
        if hasattr(Y, 'name'):
            table.title("Frequencies of %s" % Y.name)
        table.cell()
        table.cell('n')
        table.midrule()
        for cell in Y.cells:
            table.cell(cell)
            table.cell(np.sum(Y == cell))
        return table

    ct = _structure.celltable(Y, X)

    Y_categories = ct.Y.cells

    # header
    n_Y_categories = len(Y_categories)
    table = fmtxt.Table('l' * (n_Y_categories + 1))
    # header line 1
    table.cell()
    table.cell(Y.name, width=n_Y_categories, just='c')
    table.midrule(span=(2, n_Y_categories + 1))
    # header line 2
    table.cell(X.name)
    for Ycell in Y_categories:
        table.cell(Ycell)
    table.midrule()

    # body
    for cell in ct.cells:
        table.cell(ct.cell_label(cell))
        data = ct.data[cell]
        for Ycell in Y_categories:
            n = np.sum(data == Ycell)
            table.cell(n)

    # title
    if title:
        if '{Yname}' in title:
            try:
                Yname = Y.name.capitalize()
            except:
                Yname = '[unnamed]'
            title = title.format(Yname=Yname)
        table.title(title)

    return table



def stats(Y, y, x=None, match=None, sub=None, fmt='%.4g', funcs=[np.mean]):
    """
    return a table with statistics per cell.

    y : factor
        model specifying columns

    x : factor or ``None``
        model specifying rows

    funcs : list of callables
        a list of statistics functions to show (all functions must return
        scalars)


    **Example**::

        >>> A.table.stats(Y, condition, funcs=[np.mean, np.std])
        Condition   mean     std
        ----------------------------
        control     0.0512   0.08075
        test        0.2253   0.2844

    """
    Y = _data.asvar(Y)
    if x is None:
        ct = _structure.celltable(Y, y, sub=sub, match=match)

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
        ct = _structure.celltable(Y, y % x, sub=sub, match=match)

        N = len(x.cells)
        table = fmtxt.Table('l' * (N + 1))

        # table header
        table.cell()
        table.cell(x.name, width=N, just='c')
        table.midrule(span=(2, N + 1))
        table.cell()

        table.cells(*x.cells)
        table.midrule()

        # table body
        fmt_n = fmt.count('%')
        if fmt_n == 1:
            fmt_once = False
        elif len(funcs) == fmt_n:
            fmt_once = True
        else:
            raise ValueError("fmt does not match funcs")

        for Ycell in y.cells:
            table.cell(Ycell)
            for Xcell in x.cells:
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



def rm_table(Y, X=None, match=None, cov=[], sub=None, fmt='%r', labels=True,
             show_case=True):
    """
    returns a repeated-measures table


    Parameters
    ----------

    Y :
        variable to display (can be model with several dependents)

    X :
        categories defining cells (factorial model)

    match :
        factor to match values on and return repeated-measures table

    cov :
        covariate to report (WARNING: only works with match, where each value
        on the matching variable corresponds with one value in the covariate)

    sub :
        boolean array specifying which values to include (generate e.g.
        with 'sub=T==[1,2]')

    fmt :
        Format string

    labels :
        display labels for nominal variables (otherwise display codes)

    show_case : bool
        add a column with the case identity

    """
    if hasattr(Y, '_items'):  # dataframe
        Y = Y._items
    Y = _data.asmodel(Y)
    if _data.isfactor(cov) or _data.isvar(cov):
        cov = [cov]

    data = []
    names_yname = []  # names including Yi.name for matched table headers
    ynames = []  # names of Yi for independent measures table headers
    within_list = []
    for Yi in Y.effects:
        # FIXME: temporary _split_Y replacement
        ct = _structure.celltable(Yi, X, match=match, sub=sub)

        data += ct.get_data()
        names_yname += ['({c})'.format(c=n) for n in ct.cells]
        ynames.append(Yi.name)
        within_list.append(ct.all_within)
    within = within_list[0]
    assert all([w == within for w in within_list])

    # table
    n_dependents = len(Y.effects)
    n_cells = int(len(data) / n_dependents)
    if within:
        n, k = len(data[0]), len(data)
        table = fmtxt.Table('l' * (k + show_case + len(cov)))

        # header line 1
        if show_case:
            table.cell(match.name)
            case_labels = ct.matchlabels[ct.cells[0]]
            assert all(np.all(case_labels == l) for l in ct.matchlabels.cells)
        for i in range(n_dependents):
            for name in ct.cells:
                table.cell(name.replace(' ', '_'))
        for c in cov:
            table.cell(c.name)

        # header line 2
        if n_dependents > 1:
            if show_case:
                table.cell()
            for name in ynames:
                [table.cell('(%s)' % name) for i in range(n_cells)]
            for c in cov:
                table.cell()

        # body
        table.midrule()
        for i in range(n):
            case = case_labels[i]
            if show_case:
                table.cell(case)
            for j in range(k):
                table.cell(data[j][i], fmt=fmt)
            # covariates
            indexes = match == case
            for c in cov:
                # test it's all the same values
                case_cov = c[indexes]
                if len(np.unique(case_cov.x)) != 1:
                    msg = 'covariate for case "%s" has several values' % case
                    raise ValueError(msg)
                # get value
                first_i = np.nonzero(indexes)[0][0]
                cov_value = c[first_i]
                table.cell(cov_value, fmt=fmt)
    else:
        table = fmtxt.Table('l' * (1 + n_dependents))
        table.cell(X.name)
        [table.cell(y) for y in ynames]
        table.midrule()
        # data is now sorted: (cell_i within dependent_i)
        # sort data as (X-cell, dependent_i)
        data_sorted = []
        for i_cell in range(n_cells):
            data_sorted.append([data[i_dep * n_cells + i_cell] for i_dep in \
                               range(n_dependents)])
        # table
        for name, cell_data in zip(ct.cells, data_sorted):
            for i in range(len(cell_data[0])):
                table.cell(name)
                for dep_data in cell_data:
                    v = dep_data[i]
                    table.cell(v, fmt=fmt)
    return table
