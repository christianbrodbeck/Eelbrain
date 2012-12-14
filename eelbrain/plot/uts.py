'''
plotting functions for univariate uniform time-series.



Created on Mar 13, 2012

@author: christian
'''
from __future__ import division

import os

import numpy as np
import scipy.stats
import matplotlib.cm as _cm

try:
    import wx
    from wx.lib.dialogs import ScrolledMessageDialog as TextDialog
except:
    pass

from eelbrain.vessels import data as _data
from eelbrain.vessels.structure import celltable

import _base


__hide__ = ['plt', 'division', 'celltable']


class stat(_base.subplot_figure):
    def __init__(self, Y='Y', X=None, dev=scipy.stats.sem, main=np.mean,
                 sub=None, match=None, ds=None, Xax=None, ncol=3,
                 width=6, height=3, dpi=90, legend='upper right',
                 title='plot.uts.stat - {name}', figtitle=None, axtitle='{name}',
                 ylabel=True, xlabel=True, invy=False,
                 bottom=None, top=None,
                 xdim='time', cm=_cm.jet, colors=None):
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
    colors : None | list | dict
        Override the default color assignment base on a colormap ``cm``:
        **list**: map the sequence of colors on X.cells.
        **dict**: provide a color for each cell in X.


    **figure parameters:**

    legend : str | None
        matplotlib figure legend location argument
    title : str | False
        axes title; '{name}' will be formatted to ``Y.name``
    invy : bool
        invert the y axis
    bottom, top | None | scalar
        Set an absolute range for the plot's y axis.

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

        # assemble colors
        if X is None:
            if colors is None:
                colors = 'b'
            elif isinstance(colors, (list, tuple)):
                colors = colors[0]
            colors = {None: colors}
        else:
            if isinstance(colors, (list, tuple)):
                colors = dict(zip(X.cells, colors))
            if isinstance(colors, dict):
                for cell in X.cells:
                    if cell not in colors:
                        raise KeyError("%s not in colors" % repr(cell))
            else:
                cells = _data.ascategorial(X, sub=sub, ds=ds).cells
                N = len(cells)
                colors = {cell:cm(i / N) for i, cell in enumerate(cells)}

        legend_h = {}
        kwargs = dict(dev=dev, main=main, ylabel=ylabel, xdim=xdim,
                      invy=invy, bottom=bottom, top=top,
                      xlabel=xlabel, colors=colors, legend_h=legend_h)

        if '{name}' in title:
            title = title.format(name=ct.Y.name)
        super(stat, self).__init__(title=title, figsize=figsize, dpi=dpi)

        self.axes = []
        if Xax is None:
            ax = self.figure.add_subplot(111)
            self.axes.append(ax)
            if axtitle and '{name}' in axtitle:
                title_ = axtitle.format(name=ct.Y.name)
            else:
                title_ = axtitle
            _ax_stat(ax, ct, title=title_, **kwargs)
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
                title_ = axtitle.format(name=_data.cellname(cell))
                _ax_stat(ax, cct, title=title_, **kwargs)
                self.axes.append(ax)

        self.legend_handles = legend_h.values()
        self.legend_labels = legend_h.keys()
        self.plot_legend(legend)

        self._cluster_h = []
        self.cluster_info = []

        self._show(figtitle=figtitle)

    def _fill_toolbar(self, tb):
        btn = self._cluster_btn = wx.Button(tb, wx.ID_ABOUT, "Clusters")
        btn.Enable(False)
        tb.AddControl(btn)
        btn.Bind(wx.EVT_BUTTON, self._OnShowClusterInfo)

    def _OnShowClusterInfo(self, event):
        size = (350, 700)
        info = (os.linesep * 2).join(map(str, self.cluster_info))
        dlg = TextDialog(self._frame, info, "Clusters", size=size)
        dlg.ShowModal()
        dlg.Destroy()

    def plot_clusters(self, clusters, p=0.05, color=(.7, .7, .7), ax=0, clear=True):
        """Add clusters from a cluster test to the uts plot (as shaded area).

        Arguments
        ---------

        clusters : list of ndvars
            The clusters, as stored in the cluster test results
            :py:attr:`.clusters` dictionary.

        p : scalar
            Threshold p value: plot all clusters with p <= this value.

        color : matplotlib color
            Color for the cluster.

        ax : int
            Index of the axes (in the uts plot) to which the clusters are to
            be plotted.

        """
        if clear:
            for h in self._cluster_h:
                h.remove()
            self._cluster_h = []
            self.cluster_info = []

        if hasattr(clusters, 'clusters'):
            self.cluster_info.append(clusters.as_table())
            clusters = clusters.clusters
            if self._is_wx:
                self._cluster_btn.Enable(True)

        ax = self.axes[ax]
        for c in clusters:
            if c.properties['p'] <= p:
                i0 = np.nonzero(c.x)[0][0]
                i1 = np.nonzero(c.x)[0][-1]
                t0 = c.time[i0]
                t1 = c.time[i1]
                h = ax.axvspan(t0, t1, zorder= -1, color=color)
                self._cluster_h.append(h)
        self.draw()

    def plot_legend(self, loc='fig', figsize=(2, 2)):
        """Plots (or removes) the legend from the figure.

        Possible values for the ``loc`` argument:

        ``False``/``None``:
            Make the current legend invisible
        'fig':
            Plot the legend in a new figure
        str | int
            Matplotlib position argument: plot the legend on the figure


        legend content can be modified through the figure's
        ``legend_handles`` and ``legend_labels`` attributes.


        Matplotlib Position Arguments
        -----------------------------

        'upper right'  : 1,
        'upper left'   : 2,
        'lower left'   : 3,
        'lower right'  : 4,
        'right'        : 5,
        'center left'  : 6,
        'center right' : 7,
        'lower center' : 8,
        'upper center' : 9,
        'center'       : 10,

        """
        if loc and len(self.legend_handles) > 1:
            handles = self.legend_handles
            labels = self.legend_labels
            if loc == 'fig':
                return _base.legend(handles, labels, figsize=figsize)
            else:
                self.legend = self.figure.legend(handles, labels, loc=loc)
                self.draw()
        else:
            if hasattr(self, 'legend'):
                self.legend.set_visible(False)
                del self.legend
                self.draw()

    def set_ylim(self, bottom=None, top=None):
        """
        Adjust the y-axis limits on all axes (see matplotlib's
        :py:meth:`axes.set_ylim`)

        """
        for ax in self.axes:
            ax.set_ylim(bottom, top)

        self.draw()



class uts(_base.subplot_figure):
    def __init__(self, epochs, ncol=3, width=6, height=3, dpi=90,
                 title='plot.uts.uts', figtitle=None, axtitle='{name}',
                 ):
        epochs = self.epochs = _base.unpack_epochs_arg(epochs, 1)

        nplot = len(epochs)
        ncol = min(nplot, ncol)
        nrow = round(nplot / ncol + .49)
        figsize = (ncol * width, nrow * height)

        super(uts, self).__init__(title=title, figsize=figsize, dpi=dpi)

        for i, epoch in enumerate(epochs):
            ax = self.figure.add_subplot(nrow, ncol, i + 1)
            _ax_uts(ax, epoch, title=axtitle)

        self._show(figtitle=figtitle)



def _ax_stat(ax, ct, colors, legend_h={},
             dev=scipy.stats.sem, main=np.mean,
             sub=None, match=None, ds=None,
             figsize=(6, 3), dpi=90, legend='upper right', title=True, ylabel=True,
             xdim='time', xlabel=True, invy=False, bottom=None, top=None):

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
        bottom = bottom if (bottom is not None) else y1
        top = top if (top is not None) else y0
    if (bottom is not None) or (top is not None):
        ax.set_ylim(bottom, top)




class clusters(_base.subplot_figure):
    def __init__(self, epochs, pmax=0.05, ptrend=0.1, 
                 title="plot.uts.clusters", figtitle=None, axtitle='{name}',
                 cm=_cm.jet, width=6, height=3, frame=.1, dpi=90,
                  overlay=False, t = {'linestyle': 'solid', 'tcolor': 'k'}):
        
        """
        Specialized plotting function for Permutation Cluster test results

        pmax : scalar
            Maximum p-value of clusters to plot as solid.
        ptrend : scalar
            Maximum p-value of clusters to plot as trend.
        t : dict
            Plot threshold for forming clusters.
            Contains threshold plotting properties.
        linestyle : str
            Line style for threshold. 
            e.g. ['solid' | 'dashed' | 'dashdot' | 'dotted']
        color : str
            Line color for threshold.
        title : str
            Window title.
        figtitle : str | None
            Figure title.
        axtitle : str | None
            Axes title pattern. '{name}' is formatted to the first layer's
            name
        overlay : bool
            Plot epochs (time course for different effects) on top of each
            other (as opposed to on separate axes).

        """
        try:
            self.cluster_info = epochs.as_table()
            if not isinstance(self.cluster_info, (list, tuple)):
                self.cluster_info = [self.cluster_info]
        except AttributeError:
            self.cluster_info = "No Cluster Info"

        epochs = self.epochs = _base.unpack_epochs_arg(epochs, 1)

        # create figure
        N = len(epochs)
        Nax = 1 if overlay else N
        x_size = width
        y_size = height * Nax
        figsize = (x_size, y_size)

        self._caxes = []
        title = unicode(title)

        super(clusters, self).__init__(title=title, figsize=figsize, dpi=dpi)
        self.figure.subplots_adjust(hspace=.2, top=.95, bottom=.05)

        width = .85
        height = .95 / Nax
        
        for i, layers in enumerate(epochs):
            if i < Nax:  # create axes
                ax = self.figure.add_subplot(Nax, 1, i + 1)
                title_ = axtitle
            else:
                title_ = None

            # color
            color = cm(i / N)
            cax = _ax_clusters(ax, layers, color=color, pmax=pmax, 
                               title=title_, ptrend=ptrend, tkwargs = t)
            self._caxes.append(cax)

        self._show(figtitle=figtitle)


    def _fill_toolbar(self, tb):
        btn = wx.Button(tb, wx.ID_ABOUT, "Clusters")
        tb.AddControl(btn)
        btn.Bind(wx.EVT_BUTTON, self._OnShowClusterInfo)

    def _OnShowClusterInfo(self, event):
        size = (350, 700)
        info = (os.linesep * 2).join(map(str, self.cluster_info))
        dlg = TextDialog(self._frame, info, "Clusters", size=size)
        dlg.ShowModal()
        dlg.Destroy()

    def set_pmax(self, pmax=0.05):
        "set the threshold p-value for clusters to be displayed"
        for cax in self._caxes:
            cax.set_pmax(pmax=pmax)

        self.canvas.draw()




def _ax_uts(ax, layers, title=False, bottom=None, top=None, invy=False,
            xlabel=True, ylabel=True, color=None, xdim='time'):
    contours = {}
    for l in layers:
        colorspace = _base.read_cs_arg(l, None)
        if colorspace.cmap:
            _plt_uts(ax, l, color=color, xdim=xdim)
        for v, color in colorspace.contours.iteritems():
            if v in contours:
                continue
            contours[v] = ax.axhline(v, color=color)

    l0 = layers[0]
    x = l0.get_dim(xdim)
    ax.set_xlim(x[0], x[-1])
    ax.x_fmt = "t = %.3f s"

    if title:
        if 'name' in title:
            title = title.format(name=l0.name)
        ax.set_title(title)

    if xlabel:
        if xlabel is True:
            xlabel = x.name
        ax.set_xlabel(xlabel)

#    if ylabel:
#        if ylabel is True:
#            ylabel = l.properties.get('unit', None)
#        ax.set_ylabel(ylabel)

    if invy:
        y0, y1 = ax.get_ylim()
        bottom = bottom if (bottom is not None) else y1
        top = top if (top is not None) else y0
    if (bottom is not None) or (top is not None):
        ax.set_ylim(bottom, top)


def _plt_uts(ax, layer, color=None, xdim='time', kwargs={}):
    x = layer.get_dim(xdim).x
    y = layer.get_data((xdim,))
    if color is not None:
        kwargs['color'] = color
    ax.plot(x, y, **kwargs)


class _ax_clusters:
    def __init__(self, ax, layers, color=None, pmax=0.05, ptrend=0.1,
                 tkwargs = {}, xdim='time', title=None):
        Y = layers[0]

        if title:
            if '{name}' in title:
                title = title.format(name=Y.name)
            ax.set_title(title)

        if tkwargs:
            t = Y.properties.get('threshold', None)
            ax.axhline(t, **tkwargs)
        ylabel = Y.properties.get('unit', None)

        _plt_uts(ax, Y, color=color, xdim=xdim)
        if ylabel:
            ax.set_ylabel(ylabel)
        if np.any(Y.x < 0) and np.any(Y.x > 0):
            ax.axhline(0, color='k')

        self.ax = ax
        x = Y.get_dim(xdim).x
        self.xlim = (x[0], x[-1])
        self.clusters = layers[1:]
        self.cluster_hs = {}
        self.sig_kwargs = dict(color=color, xdim=xdim, y=Y)
        self.trend_kwargs = dict(color=(.7, .7, .7), xdim=xdim, y=Y)
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
#        ax.set_ylim(bottom=0)

    def set_pmax(self, pmax=0.05, ptrend=0.1):
        self.pmax = pmax
        self.ptrend = ptrend
        self.draw()


def _plt_cluster(ax, ndvar, color=None, y=None, xdim='time',
                 hatch='/'):
    x = ndvar.get_dim(xdim).x
    v = ndvar.get_data((xdim,))
    where = np.nonzero(v)[0]

    # make sure the cluster is contiguous
    if len(where) > 1 and np.max(np.diff(where)) > 1:
        raise ValueError("Non-contiguous clusters not supported; where = %r" % where)

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


