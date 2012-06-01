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



def frequencies(Y, X=None, sub=None, title="{Yname} Frequencies", ds=None):
    """
    Display frequency of occurrence of all categories in Y in the cells 
    defined by X.
    
    Y: factor whose frequencies are of interest
    X: model defining cells for which frequencies are displayed
    
    """
    if isinstance(X, str):
        X = ds[X]
    if isinstance(Y, str):
        Y = ds[Y]
    if isinstance(sub, str):
        sub = ds[sub]
    
    Y = _data.asfactor(Y)
    
    if X is None:
        table = fmtxt.Table('ll')
        if hasattr(Y, 'name'):
            table.title("Frequencies of %s" % Y.name)
        table.cell()
        table.cell('n')
        table.midrule()
        for cat in Y.values():
            table.cell(cat)
            table.cell(np.sum(Y == cat))
        return table
    
    
    X = _data.asfactor(X)
    
    ct = _structure.celltable(Y, X, sub=sub)
    
    Y_categories = ct.Y.values()
    
    # header
    n_Y_categories = len(Y_categories)
    table = fmtxt.Table('l' * (n_Y_categories+1))
    # header line 1
    table.cell()
    table.cell(Y.name, width=n_Y_categories, just='c')
    table.midrule(span=(2, n_Y_categories+1))
    # header line 2
    table.cell(X.name)
    for Ycell in Y_categories:
        table.cell(Ycell)
    table.midrule()
    
    # body
    for cell in ct.cells:
        table.cell(cell)
        data = ct.data[cell]
        for Ycell in Y_categories:
            n = np.sum(data == Ycell)
            table.cell(n)
    
    # title
    if title:
        title = title.format(Yname=Y.name.capitalize())
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
    y_ids = sorted(y.cells.keys())
    
    if x is None:
        ct = _structure.celltable(Y, y, sub=sub, match=match)
        
        # table header
        n_disp = len(funcs)
        table = fmtxt.Table('l'*(n_disp+1))
        table.cell('Condition', 'bf')
        for func in funcs:
            table.cell(func.__name__, 'bf')
        table.midrule()
        
        # table entries
        for cell in ct.cells:
            data = ct.data[cell]
            table.cell(cell)
            for func in funcs:
                table.cell(fmt % func(data))
    else:
        ct = _structure.celltable(Y, y%x, sub=sub, match=match)
        
        table = fmtxt.Table('l'*(len(x.cells)+1))
        x_ids = sorted(x.cells.keys())
        
        # table header
        table.cell()
        table.cell(x.name, width=len(x_ids), just='c')
        table.midrule(span=(2, 1+len(x_ids)))
        table.cell()
        
        for xid in x_ids:
            table.cell(x.cells[xid])
        table.midrule()
        
        # table body
        fmt_n = fmt.count('%')
        if fmt_n == 1:
            fmt_once = False
        elif len(funcs) == fmt_n:
            fmt_once = True
        else:
            raise ValueError("fmt does not match funcs")
        
        for Ycell in y.values():
            table.cell(Ycell)
            for Xcell in x.values():
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
                values = (f(data) for f in funcs)
                if fmt_once:
                    txt = fmt % values
                else:
                    txt = ', '.join((fmt % v for v in values))
                
                table.cell(txt)
    
    return table
        

'''
def aslongtable(*Y, **kwargs):
    "astable should be able to do everything"
    if 'sub' in kwargs:
        sub = kwargs['sub']
        Y = [y[sub] for y in Y]
    table = fmtxt.Table('l'*len(Y))
    for y in Y:
        table.cell(y.name)
    table.midrule()
    fac = [type(y) == factor for y in Y]
    Y = [y.x for y in Y]
    for line in zip(*Y):
        for v, fac_status in zip(line, fac):
            if fac_status: d = 0
            else: d = 5
            table.cell(v, digits=d)
    return table
'''


def rm_table(Y, X=None, match=None, cov=[], sub=None, fmt='%r', labels=True, 
             show_case=True):
    """
    returns a repeated-measures table
    
    **parameters:**

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
    if hasattr(Y, '_items'): # dataframe
        Y = Y._items
    Y = _data.asmodel(Y)
    if _data.isfactor(cov) or _data.isvar(cov):
        cov = [cov]
    
    data = []
    names_yname = [] # names including Yi.name for matched table headers
    ynames = [] # names of Yi for independent measures table headers
    within_list = []
    for Yi in Y.effects:
        # FIXME: temporary _split_Y replacement 
        ct = _structure.celltable(Yi, X, match=match, sub=sub)
        
        data += ct.get_data()
        names_yname += ['({c})'.format(c=n) for n in ct.cells]
        ynames.append(Yi.name)
        within_list.append(ct.all_within)
    within = within_list[0]
    assert all([w==within for w in within_list])
    
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
            assert all(np.all(case_labels == l) for l in ct.matchlabels.values())
        for i in range(n_dependents):
            for name in ct.cells:        
                table.cell(name.replace(' ','_'))
        for c in cov:
            table.cell(c.name)
        
        # header line 2
        if n_dependents > 1:
            if show_case:
                table.cell()
            for name in ynames:
                [table.cell('(%s)'%name) for i in range(n_cells)]
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
            indexes = match==case
            for c in cov:
                # test it's all the same values
                case_cov = c[indexes]
                if len(np.unique(case_cov.x)) != 1: 
                    msg = 'covariate for case "%s" has several values'%case
                    raise ValueError(msg)
                # get value
                first_i = np.nonzero(indexes)[0][0]
                cov_value = c[first_i]
                if _data.isfactor(c) and labels:
                    cov_value = c.cells[cov_value]
                table.cell(cov_value, fmt=fmt)
    else:
        table = fmtxt.Table('l'*(1 + n_dependents))
        table.cell(X.name)
        [table.cell(y) for y in ynames]
        table.midrule()
        # data is now sorted: (cell_i within dependent_i)
        # sort data as (X-cell, dependent_i)
        data_sorted = []
        for i_cell in range(n_cells):
            data_sorted.append([data[i_dep*n_cells + i_cell] for i_dep in \
                               range(n_dependents)])
        # table
        for name, cell_data in zip(ct.cells, data_sorted):
            for i in range(len(cell_data[0])):
                table.cell(name)
                for dep_data in cell_data:
                    v = dep_data[i]
                    table.cell(v, fmt=fmt)
    return table


