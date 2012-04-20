'''
plottind functions for 1-dimensional uniform time-series.



Created on Mar 13, 2012

@author: christian
'''

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

import eelbrain.vessels.data as _data


__hide__ = ['plt']


def uts(var='Y', conditions=None, ct=np.mean, dev=scipy.stats.sem, ds=None,
        figsize=(6,3), dpi=90):
    """
    Plots a one-dimensional ndvar
    
    var : ndvar
        dpendent variable (one-dimensiona ndvar)
    conditions : categorial variable or None
        conditions which should be plotted separately
    ct : func
        central tendency (function that takes an ``axis`` argument)
    dev : str
        Measure for deviation: 
        ``None``: no statistics
        ``'all'``: plot all individual traces
    
    if ``var`` or ``conditions`` is submitted as string, the ``ds`` argument 
    must provide a dataset containing those variables.
    
    """
    if isinstance(var, basestring):
        var = ds[var]
    if isinstance(conditions, basestring):
        conditions = ds[conditions]
    
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.subplots_adjust(.1, .2, .95, .95, .1, .4)
    ax = plt.axes()
    
    if conditions:
        conditions = _data.asfactor(conditions)
        labels = conditions.cells.values()
        groups = [var[conditions==c] for c in labels]
    else:
        labels = [None]
        groups = [var]
    
    h = []
    for group, lbl in zip(groups, labels):
        _h = _plt_uts(ax, group, ct, dev, label=lbl)
        h.append(_h)
    
    dim = var.dims[0]
    ax.set_xlabel(dim.name)
    ax.set_xlim(dim.x[0], dim.x[-1])
    ax.legend()
    
    fig.show()
    return fig



def _plt_uts(ax, ndvar, ct, dev, label=None, **kwargs):
    h = {}
    x = ndvar.dims[0].x
    
    ct_kwargs = kwargs.copy()
    if label:
        ct_kwargs['label'] = label
    
    if ct == 'all':
        h['ct'] = ax.plot(x, ndvar.x.T, **ct_kwargs)
        dev = None
    elif hasattr(ct, '__call__'):
        y = ct(ndvar.x, axis=0)
        h['ct'] = ax.plot(x, y, **ct_kwargs)
    else:
        raise ValueError("Invalid argument: ct=%r" % ct)
    
    kwargs['alpha'] = .2
    if hasattr(dev, '__call__'):
        ydev = dev(ndvar.x, axis=0)
        h['dev'] = ax.fill_between(x, y-ydev, y+ydev, **kwargs)
    else:
        pass
    
    return h
        
