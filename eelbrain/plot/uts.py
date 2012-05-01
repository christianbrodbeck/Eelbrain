'''
plottind functions for 1-dimensional uniform time-series.



Created on Mar 13, 2012

@author: christian
'''
from __future__ import division


import numpy as np
import scipy.stats
import matplotlib.cm as _cm
import matplotlib.pyplot as plt

import eelbrain.vessels.data as _data


__hide__ = ['plt', 'division']


def stat(var='Y', X=None, main=np.mean, dev=scipy.stats.sem, ds=None,
         figsize=(6,3), dpi=90, legend='upper right', title=True,
         xdim='time', cm=_cm.jet):
    """
    Plots statistics for a one-dimensional ndvar
    
    var : ndvar
        dependent variable (one-dimensional ndvar)
    X : categorial or None
        conditions which should be plotted separately
    main : func | 'all' | float
        central tendency (function that takes an ``axis`` argument). For float
        or 'all', ``dev = None``
    dev : func | ``None``
        Measure for spread / deviation from the central tendency (function 
        that takes an ``axis`` argument)
    ds : dataset
        if ``var`` or ``X`` is submitted as string, the ``ds`` argument 
        must provide a dataset containing those variables.
    
    **plotting parameters:**
    
    xdim : str
        dimension for the x-axis (default is 'time') 
    cm : matplotlib colormap
        colormap from which colors for different categories in ``X`` are 
        derived
    
    **figure parameters:**
    
    legend : str | None
        matplotlib figure legend location argument
    title : str | True | False
        axes title; if ``True``, use ``var.name``
    
    """
    if isinstance(var, basestring):
        var = ds[var]
    if title is True:
        title = var.name
    if isinstance(X, basestring):
        X = ds[X]
        
    
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    top = .9 # - bool(title) * .1
    plt.subplots_adjust(.1, .1, .95, top, .1, .4)
    ax = plt.axes()
    
    if X:
        X = _data.ascategorial(X)
        values = X.values()
        groups = [var[X==v] for v in values]
    else:
        values = [None]
        groups = [var]
    
    h = []
    legend_h = []
    legend_lbl = []
    N = len(groups)
    for i, group, value in zip(range(N), groups, values):
        lbl = str(value)
        c = cm(i / N)
        _h = _plt_stat(ax, group, main, dev, label=lbl, xdim=xdim, color=c)
        h.append(_h)
        legend_h.append(_h['main'][0])
        legend_lbl.append(lbl)
    
    dim = var.get_dim(xdim)
    if title:
        ax.set_title(title)
    ax.set_xlabel(dim.name)
    ax.set_xlim(min(dim), max(dim))
    if legend and any(values):
        fig.legend(legend_h, legend_lbl, loc=legend)
    
    fig.show()
    return fig



def _plt_stat(ax, ndvar, main, dev, label=None, xdim='time', color=None, **kwargs):
    h = {}
    dim = ndvar.get_dim(xdim)
    x = dim.x
    y = ndvar.get_data(('epoch', 'time'))
    
    main_kwargs = kwargs.copy()
    if label:
        main_kwargs['label'] = label
    if color:
        main_kwargs['color'] = color
        kwargs['alpha'] = .3
        kwargs['color'] = color
    if isinstance(main, float):
        main_kwargs['alpha'] = main
        main = 'all'
    
    if main == 'all':
        h['main'] = ax.plot(x, y.T, **main_kwargs)
        dev = None
    elif hasattr(main, '__call__'):
        y_ct = main(y, axis=0)
        h['main'] = ax.plot(x, y_ct, **main_kwargs)
    else:
        raise ValueError("Invalid argument: main=%r" % main)
    
    if hasattr(dev, '__call__'):
        ydev = dev(y, axis=0)
        h['dev'] = ax.fill_between(x, y_ct-ydev, y_ct+ydev, **kwargs)
    else:
        pass
    
    return h
        
