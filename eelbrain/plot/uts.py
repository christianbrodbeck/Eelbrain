'''
plotting functions for univariate uniform time-series.



Created on Mar 13, 2012

@author: christian
'''
from __future__ import division


import numpy as np
import scipy.stats
import matplotlib.cm as _cm
import matplotlib.pyplot as plt

from eelbrain.vessels import data as _data
from eelbrain.vessels.structure import celltable
from eelbrain.wxutils import mpl_canvas

import _base


__hide__ = ['plt', 'division', 'celltable']


class stat(mpl_canvas.CanvasFrame):
    def __init__(self, Y='Y', X=None, dev=scipy.stats.sem, main=np.mean,
                 sub=None, match=None, ds=None, Xax=None, ncol=3,
                 width=6, height=3, dpi=90, legend='upper right', title=True,
                 ylabel=True, xlabel=True, invy=False,
                 xdim='time', cm=_cm.jet):
        """
    Plots statistics for a one-dimensional ndvar

    Y : 1d-ndvar
        dependent variable (one-dimensional ndvar)
    X : categorial or None
        conditions which should be plotted separately
    main : func | None
        central tendency (function that takes an ``axis`` argument).
    dev : func | 'all' | float
        Measure for spread / deviation from the central tendency. Either a
        function that takes an ``axis`` argument, 'all' to plot all traces, or
        a float to plot all traces with a certain alpha value
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
    invy : bool
        invert the y axis

        """
        if Xax is None:
            figsize = (width, height)
            ct = celltable(Y, X, sub=sub, match=match, ds=ds)
        else:
            ct = celltable(Y, Xax, sub=sub, ds=ds)
            if X is not None:
                Xct = celltable(X, Xax, sub=sub, ds=ds)
            if match is not None:
                matchct = celltable(match, Xax, sub=sub, ds=ds)
            nplot = len(ct.cells)
            ncol = min(nplot, ncol)
            nrow = round(nplot / ncol + .49)
            figsize = (ncol * width, nrow * height)

        if X is None:
            colors = {ct.Y.name: 'b'}
        else:
            cells = _data.ascategorial(X, sub=sub, ds=ds).cells
            N = len(cells)
            colors = {cell:cm(i / N) for i, cell in enumerate(cells)}

        legend_h = {}
        kwargs = dict(dev=dev, main=main, ylabel=ylabel, xdim=xdim, invy=invy,
                      xlabel=xlabel, colors=colors, legend_h=legend_h)

        if title is True:
            title = ct.Y.name

        if isinstance(title, basestring):
            win_title = title
        else:
            win_title = ct.Y.name
        super(stat, self).__init__(title=win_title, figsize=figsize, dpi=dpi)

        self.axes = []
        if Xax is None:
#            top = .9 # - bool(title) * .1
#        plt.subplots_adjust(.1, .1, .95, top, .1, .4)
#            ax = self.figure.add_axes([.1, .1, .8, .8])
            ax = self.figure.add_subplot(111)
            self.axes.append(ax)
            _ax_stat(ax, ct, title=title, **kwargs)
            if len(ct) < 2:
                legend = False
        else:
            for i, cell in enumerate(ct.cells):
                kwargs['xlabel'] = xlabel if i == len(ct) - 1 else False
                ax = self.figure.add_subplot(nrow, ncol, i + 1)
                if X is not None:
                    X = Xct.data[cell]
                if match is not None:
                    match = matchct.data[cell]
                cct = celltable(ct.data[cell], X, match=match)
                _ax_stat(ax, cct, title=_data.cellname(cell), ** kwargs)
                self.axes.append(ax)

        if legend and len(legend_h) > 1:
            self.figure.legend(legend_h.values(), legend_h.keys(), loc=legend)

        self.figure.tight_layout()
        self.Show()



def _ax_stat(ax, ct, colors, legend_h={},
             dev=scipy.stats.sem, main=np.mean,
             sub=None, match=None, ds=None,
             figsize=(6, 3), dpi=90, legend='upper right', title=True, ylabel=True,
             xdim='time', xlabel=True, invy=False):

    ax.x_fmt = "t = %.3f s"

    h = []
    for cell in ct.cells:
        lbl = ct.cell_label(cell, ' ')
        c = colors[cell]
        _h = _plt_stat(ax, ct.data[cell], main, dev, label=lbl, xdim=xdim, color=c)
        h.append(_h)
        if lbl not in legend_h:
            legend_h[lbl] = _h['main'][0]

    dim = ct.Y.get_dim(xdim)

    if title:
        if title is True:
            title = ct.Y.name
        ax.set_title(title)

    if xlabel:
        if xlabel is True:
            xlabel = dim.name
        ax.set_xlabel(xlabel)
    ax.set_xlim(min(dim), max(dim))

    if ylabel is True:
        ylabel = ct.Y.properties.get('unit', None)
    if ylabel:
        ax.set_ylabel(ylabel)
    if invy:
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y1, y0)



class clusters(mpl_canvas.CanvasFrame):
    def __init__(self, epochs, pmax=0.05, ptrend=0.1, t=True,
                 title=None, cm=_cm.jet,
                 width=6, height=3, frame=.1, dpi=90,
                 overlay=False):
        """
        Specialized plotting function for Permutation Cluster test results

        t : bool
            plot threshold

        """
        epochs = self.epochs = _base.unpack_epochs_arg(epochs, 1)

        # create figure
        N = len(epochs)
        x_size = width
        y_size = height if overlay else height * N
        figsize = (x_size, y_size)

        self._caxes = []
        if title:
            title = unicode(title)
        else:
            title = ''

        super(clusters, self).__init__(title=title, figsize=figsize, dpi=dpi)
        self.figure.subplots_adjust(hspace=.2, top=.95, bottom=.05)

        width = .85
        if overlay:
            height = .95
            ax = self.figure.add_subplot(111)
        else:
            height = .95 / N

        for i, layers in enumerate(epochs):
            if not overlay: # create axes
                ax = self.figure.add_subplot(N, 1, i + 1)
                ax.set_title(layers[0].name)

            # color
            color = cm(i / N)
            cax = _ax_clusters(ax, layers, color=color, pmax=pmax, ptrend=ptrend, t=t)
            self._caxes.append(cax)

        self.figure.tight_layout()
        self.Show()

    def set_pmax(self, pmax=0.05):
        "set the threshold p-value for clusters to be displayed"
        for cax in self._caxes:
            cax.set_pmax(pmax=pmax)

        self.canvas.draw()




def _ax_uts(ax, layers, color=None, xdim='time'):
    for l in layers:
        _plt_uts(ax, l, color=color, xdim=xdim)

    x = layers[0].get_dim(xdim).x
    ax.set_xlim(x[0], x[-1])


def _plt_uts(ax, layer, color=None, xdim='time'):
    x = layer.get_dim(xdim).x
    y = layer.get_data((xdim,))
    ax.plot(x, y, color=color)


class _ax_clusters:
    def __init__(self, ax, layers, color=None, pmax=0.05, ptrend=0.1,
                 t=True, xdim='time'):
        Y = layers[0]
        if t is True:
            t = layers[0].properties.get('tF', None)
        if t:
            ax.axhline(t, color='k')
        ylabel = Y.properties.get('unit', None)

        _plt_uts(ax, Y, color=color, xdim=xdim)
        if ylabel:
            ax.set_ylabel(ylabel)

        self.ax = ax
        x = layers[0].get_dim(xdim).x
        self.xlim = (x[0], x[-1])
        self.clusters = layers[1:]
        self.cluster_hs = {}
        self.sig_kwargs = dict(color=color, xdim=xdim, y=layers[0])
        self.trend_kwargs = dict(color=(.7, .7, .7), xdim=xdim, y=layers[0])
        self.set_pmax(pmax=pmax, ptrend=ptrend)

    def draw(self):
        ax = self.ax

        for c in self.clusters:
            if c.properties['p'] <= self.pmax:
                if c not in self.cluster_hs:
                    h = _plt_cluster(ax, c, **self.sig_kwargs)
                    self.cluster_hs[c] = h
            elif c.properties['p'] <= self.ptrend:
                if c not in self.cluster_hs:
                    h = _plt_cluster(ax, c, **self.trend_kwargs)
                    self.cluster_hs[c] = h
            else:
                if c in self.cluster_hs:
                    h = self.cluster_hs.pop(c)
                    h.remove()

        ax.set_xlim(*self.xlim)
        ax.set_ylim(bottom=0)

    def set_pmax(self, pmax=0.05, ptrend=0.1):
        self.pmax = pmax
        self.ptrend = ptrend
        self.draw()


def _plt_cluster(ax, ndvar, color=None, y=None, xdim='time',
                 hatch='/'):
    x = ndvar.get_dim(xdim).x
    v = ndvar.get_data((xdim,))
    where = np.where(v)[0]
    assert np.abs(np.diff(where)).sum() <= 2
    x0 = where[0]
    x1 = where[-1]

    if y is None:
        h = ax.vspan(x0, x1, color=color, hatch=hatch, fill=False)
    else:
        y = y.get_data((xdim,))
        h = ax.fill_between(x, y, where=v, color=color, alpha=0.5)

    return h


def _plt_stat(ax, ndvar, main, dev, label=None, xdim='time', color=None, **kwargs):
    h = {}
    dim = ndvar.get_dim(xdim)
    x = dim.x
    y = ndvar.get_data(('case', 'time'))

    if color:
        kwargs['color'] = color

    main_kwargs = kwargs.copy()
    dev_kwargs = kwargs.copy()
    if label:
        main_kwargs['label'] = label

    if np.isscalar(dev):
        dev_kwargs['alpha'] = dev
        dev = 'all'
    else:
        dev_kwargs['alpha'] = .3

    if dev == 'all':
        if 'linewidth' in kwargs:
            main_kwargs['linewidth'] = kwargs['linewidth'] * 2
        elif 'lw' in kwargs:
            main_kwargs['lw'] = kwargs['lw'] * 2
        else:
            main_kwargs['lw'] = 2

    # plot main
    if hasattr(main, '__call__'):
        y_ct = main(y, axis=0)
        h['main'] = ax.plot(x, y_ct, zorder=5, **main_kwargs)
    elif dev == 'all':
        pass
    else:
        raise ValueError("Invalid argument: main=%r" % main)

    # plot dev
    if hasattr(dev, '__call__'):
        ydev = dev(y, axis=0)
        h['dev'] = ax.fill_between(x, y_ct - ydev, y_ct + ydev, zorder=0, **dev_kwargs)
    elif dev == 'all':
        h['dev'] = ax.plot(x, y.T, **dev_kwargs)
        dev = None
    else:
        pass

    return h


