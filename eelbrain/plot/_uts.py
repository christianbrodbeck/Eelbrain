"""Plot uniform time-series of one variable."""
# author: Christian Brodbeck
from __future__ import division

import operator
from warnings import warn

import matplotlib as mpl
import numpy as np

from .._data_obj import ascategorial, asndvar, assub, cellname, Celltable
from .._stats import stats
from . import _base
from ._base import _EelFigure, LegendMixin
from ._colors import colors_for_oneway, find_cell_colors


class UTSStat(_EelFigure, LegendMixin):
    "Plots statistics for a one-dimensional NDVar"
    def __init__(self, Y='Y', X=None, Xax=None, match=None, sub=None, ds=None,
                 main=np.mean, error='sem', pool_error=None, legend='upper right',
                 axtitle='{name}', xlabel=True, ylabel=True, invy=False,
                 bottom=None, top=None, hline=None, xdim='time', xlim=None,
                 color='b', colors=None, clusters=None, pmax=0.05,
                 ptrend=0.1, *args, **kwargs):
        """
    Plot statistics for a one-dimensional NDVar

    Parameters
    ----------
    Y : 1d-NDVar
        Dependent variable (one-dimensional NDVar).
    X : categorial or None
        Model: specification of conditions which should be plotted separately.
    Xax : None | categorial
        Make separate axes for each category in this categorial model.
    match : Factor
        Identifier for repeated measures data.
    sub : None | index array
        Only use a subset of the data provided.
    ds : None | Dataset
        If a Dataset is specified, all data-objects can be specified as
        names of Dataset variables.
    main : func | None
        Measure for the central tendency (function that takes an ``axis``
        argument). The default is numpy.mean.
    error : None | str
        Measure of variability to plot (default: 1 SEM). Examples:
        'ci': 95% confidence interval;
        '99%ci': 99% confidence interval (default);
        '2sem': 2 standard error of the mean;
        'all': plot all traces.
    pool_error : bool
        Pool the errors for the estimate of variability (default is True
        for related measures designs, False otherwise). See Loftus & Masson
        (1994).
    legend : str | int | 'fig' | None
        Matplotlib figure legend location argument or 'fig' to plot the
        legend in a separate figure.
    axtitle : str | None
        Axes title, '{name}' is formatted to the category name. When plotting
        only one axes, use the `title` argument.
    xlabel, ylabel : True |str | None
        X- and y axis label. If True the labels will be inferred from the data.
    invy : bool
        invert the y axis
    bottom, top | None | scalar
        Set an absolute range for the plot's y axis.
    hline : None | scalar | (value, kwarg-dict) tuple
        Add a horizontal line to each plot. If provided as a tuple, the second
        element can include any keyword arguments that should be submitted to
        the call to matplotlib axhline call.
    xdim : str
        dimension for the x-axis (default is 'time')
    xlim : None | (scalar, scalar)
        Tuple of xmin and xmax to set the initial x-axis limits.
    color : matplotlib color
        Color if just a single category of data is plotted.
    colors : str | list | dict
        Colors for the plots if multiple categories of data are plotted.
        **str**: A colormap name; Cells of X are mapped onto the colormap in
        regular intervals.
        **list**: A list of colors in the same sequence as X.cells.
        **dict**: A dictionary mapping each cell in X to a color.
        Colors are specified as `matplotlib compatible color arguments
        <http://matplotlib.org/api/colors_api.html>`_.
    clusters : None | Dataset
        Clusters to add to the plots. The clusters should be provided as
        Dataset, as stored in test results' :py:attr:`.clusters`.
    pmax : scalar
        Maximum p-value of clusters to plot as solid.
    ptrend : scalar
        Maximum p-value of clusters to plot as trend.
    tight : bool
        Use matplotlib's tight_layout to expand all axes to fill the figure
        (default True)
    title : str | None
        Figure title.
    frame : bool
        Draw a frame containing the figure from the top and the right (default
        ``True``).
        """
        if 'dev' in kwargs:
            error = kwargs.pop('dev')
            warn("The 'dev' keyword is deprecated, use 'error'",
                 DeprecationWarning)

        # coerce input variables
        sub = assub(sub, ds)
        Y = asndvar(Y, sub, ds)
        if X is not None:
            X = ascategorial(X, sub, ds)
        if Xax is not None:
            Xax = ascategorial(Xax, sub, ds)
        if match is not None:
            match = ascategorial(match, sub, ds)

        if pool_error or (pool_error is None and match is not None):
            all_x = [i for i in (Xax, X) if i is not None]
            if len(all_x) > 0:
                full_x = reduce(operator.mod, all_x)
                ct = Celltable(Y, full_x, match)
                dev_data = stats.variability(ct.Y.x, ct.X, ct.match, error, True)
                error = 'data'
            else:
                dev_data = None
        else:
            dev_data = None

        if Xax is None:
            nax = 1
            ct = Celltable(Y, X, match)
            if X is None:
                color_x = None
            else:
                color_x = ct.X
        else:
            ct = Celltable(Y, Xax)
            if X is None:
                color_x = None
                X_ = None
            else:
                Xct = Celltable(X, Xax)
                color_x = Xct.Y
            if match is not None:
                matchct = Celltable(match, Xax)
            nax = len(ct.cells)

        # assemble colors
        if color_x is None:
            colors = {None: color}
        else:
            colors = find_cell_colors(color_x, colors)

        frame_title = _base.frame_title("UTSStat", Y, X, Xax)
        _EelFigure.__init__(self, frame_title, nax, 4, 2, *args, **kwargs)

        # create plots
        self._plots = []
        legend_handles = {}
        if Xax is None:
            ax = self._axes[0]
            p = _ax_uts_stat(ax, ct, colors, main, error, dev_data, None,
                             ylabel, xdim, xlim, xlabel, invy, bottom, top,
                             hline, clusters, pmax, ptrend)
            self._plots.append(p)
            legend_handles.update(p.legend_handles)
            if len(ct) < 2:
                legend = False
        else:
            for i, ax, cell in zip(xrange(nax), self._axes, ct.cells):
                if i == len(ct) - 1:
                    xlabel_ = xlabel
                else:
                    xlabel_ = False

                if X is not None:
                    X_ = Xct.data[cell]

                if match is not None:
                    match = matchct.data[cell]

                ct_ = Celltable(ct.data[cell], X_, match=match, coercion=asndvar)
                title_ = axtitle.format(name=cellname(cell))
                p = _ax_uts_stat(ax, ct_, colors, main, error, dev_data, title_,
                                 ylabel, xdim, xlim, xlabel_, invy, bottom, top,
                                 hline, clusters, pmax, ptrend)
                self._plots.append(p)
                legend_handles.update(p.legend_handles)

        LegendMixin.__init__(self, legend, legend_handles)
        self._update_ui_cluster_button()
        self._show()

    def _fill_toolbar(self, tb):
        import wx

        btn = self._cluster_btn = wx.Button(tb, wx.ID_ABOUT, "Clusters")
        btn.Enable(False)
        tb.AddControl(btn)
        btn.Bind(wx.EVT_BUTTON, self._OnShowClusterInfo)

        LegendMixin._fill_toolbar(self, tb)

    def _OnShowClusterInfo(self, event):
        from .._wxutils import show_text_dialog

        if len(self._plots) == 1:
            clusters = self._plots[0].cluster_plt.clusters
            all_plots_same = True
        else:
            all_clusters = [p.cluster_plt.clusters is None for p in self._plots]
            clusters = all_clusters[0]
            if all(c is clusters for c in all_clusters[1:]):
                all_plots_same = True
            else:
                all_plots_same = False

        if all_plots_same:
            info = str(clusters)
        else:
            info = []
            for i, clusters in enumerate(all_clusters):
                if clusters is None:
                    continue
                title = "Axes %i" % i
                info.append(title)
                info.append('\n')
                info.append('-' * len(title))
                info.append(str(clusters))
            info = '\n'.join(info)

        show_text_dialog(self._frame, info, "Clusters")

    def _update_ui_cluster_button(self):
        if hasattr(self, '_cluster_btn'):
            enable = not all(p.cluster_plt.clusters is None for p in self._plots)
            self._cluster_btn.Enable(enable)

    def set_clusters(self, clusters, pmax=0.05, ptrend=0.1, color='.7', ax=None):
        """Add clusters from a cluster test to the plot (as shaded area).

        Parameters
        ----------
        clusters : None | Dataset
            The clusters, as stored in test results' :py:attr:`.clusters`.
            Use ``None`` to remove the clusters plotted on a given axis.
        pmax : scalar
            Maximum p-value of clusters to plot as solid.
        ptrend : scalar
            Maximum p-value of clusters to plot as trend.
        color : matplotlib color
            Color for the clusters.
        ax : None | int
            Index of the axes to which the clusters are to be added. If None,
            add the clusters to all axes.
        """
        nax = len(self._axes)
        if ax is None:
            axes = xrange(nax)
        else:
            axes = [ax]

        # update plots
        for ax in axes:
            p = self._plots[ax].cluster_plt
            p.set_clusters(clusters, False)
            p.set_color(color, False)
            p.set_pmax(pmax, ptrend)
        self.draw()

        self._update_ui_cluster_button()

    def set_xlim(self, xmin, xmax):
        "Adjust the x-axis limits on all axes"
        for ax in self._axes:
            ax.set_xlim(xmin, xmax)
        self.draw()

    def set_ylim(self, bottom=None, top=None):
        "Adjust the y-axis limits on all axes"
        for ax in self._axes:
            ax.set_ylim(bottom, top)
        self.draw()


class UTS(_EelFigure):
    "Value by time plot for UTS data."
    def __init__(self, epochs, Xax=None, axtitle='{name}', ds=None, *args, **kwargs):
        """Value by time plot for UTS data

        Parameters
        ----------
        epochs : epochs
            Uts data epochs to plot.
        Xax : None | categorial
            Make separate axes for each category in this categorial model.
        axtitle : None | str
            Axes title. '{name}' is formatted to the category name.
        ds : None | Dataset
            If a Dataset is specified, all data-objects can be specified as
            names of Dataset variables.
        tight : bool
            Use matplotlib's tight_layout to expand all axes to fill the figure
            (default True)
        title : None | str
            Figure title.
        """
        epochs = self.epochs = _base.unpack_epochs_arg(epochs, 1, Xax, ds)
        _EelFigure.__init__(self, "UTS", len(epochs), 2, 1.5, *args, **kwargs)

        for ax, epoch in zip(self._axes, epochs):
            _ax_uts(ax, epoch, title=axtitle)

        self._show()


class _ax_uts_stat:

    def __init__(self, ax, ct, colors, main, error, dev_data, title, ylabel, xdim,
                 xlim, xlabel, invy, bottom, top, hline, clusters, pmax, ptrend):
        ax.x_fmt = "t = %.3f s"

        # stat plots
        self.stat_plots = []
        self.legend_handles = {}

        x = ct.Y.get_dim(xdim)
        for cell in ct.cells:
            ndvar = ct.data[cell]
            y = ndvar.get_data(('case', xdim))
            plt = _plt_uts_stat(ax, x, y, main, error, dev_data, colors[cell],
                                cellname(cell))
            self.stat_plots.append(plt)
            if plt.main is not None:
                self.legend_handles[cell] = plt.main[0]

        # hline
        if hline is not None:
            if isinstance(hline, tuple):
                if len(hline) != 2:
                    raise ValueError("hline must be None, scalar or length 2 tuple")
                hline, hline_kw = hline
                hline_kw = dict(hline_kw)
            else:
                hline_kw = {'color': 'k'}

            hline = float(hline)
            ax.axhline(hline, **hline_kw)

        # cluster plot
        self.cluster_plt = _plt_uts_clusters(ax, clusters, pmax, ptrend)

        # title
        if title:
            if title is True:
                title = ct.Y.name
            ax.set_title(title)

        # format x axis
        if xlim is None:
            ax.set_xlim(x[0], x[-1])
        else:
            xmin, xmax = xlim
            ax.set_xlim(xmin, xmax)
        ax.x_fmt = "t = %.3f s"
        ticks = ax.xaxis.get_ticklocs()
        ticklabels = _base._ticklabels(ticks, 'time')
        ax.xaxis.set_ticklabels(ticklabels)

        _base.set_xlabel(ax, xdim, xlabel)

        if ylabel is True:
            ylabel = ct.Y.info.get('unit', None)
        if ylabel:
            ax.set_ylabel(ylabel)

        if invy:
            y0, y1 = ax.get_ylim()
            bottom = bottom if (bottom is not None) else y1
            top = top if (top is not None) else y0
        if (bottom is not None) or (top is not None):
            ax.set_ylim(bottom, top)

        # store attributes
        self.ax = ax
        self.title = title


class UTSClusters(_EelFigure):
    "Plotting of ANOVA permutation cluster test results"
    def __init__(self, res, pmax=0.05, ptrend=0.1, axtitle='{name}', cm='jet',
                 overlay=False, *args, **kwargs):
        """
        Plotting of permutation cluster test results

        Parameters
        ----------
        res : testnd.anova
            ANOVA with permutation cluster test result object.
        pmax : scalar
            Maximum p-value of clusters to plot as solid.
        ptrend : scalar
            Maximum p-value of clusters to plot as trend.
        axtitle : None | str
            Axes title pattern. '{name}' is formatted to the effect name.
        cm : str
            Colormap to use for coloring different effects.
        overlay : bool
            Plot epochs (time course for different effects) on top of each
            other (as opposed to on separate axes).
        tight : bool
            Use matplotlib's tight_layout to expand all axes to fill the figure
            (default True)
        title : str
            Figure title.
        """
        clusters_ = res.clusters

        epochs = self.epochs = _base.unpack_epochs_arg(res, 1)

        # create figure
        n = len(epochs)
        nax = 1 if overlay else n
        _EelFigure.__init__(self, "UTSClusters", nax, 4, 2, *args, **kwargs)

        colors = colors_for_oneway(range(n), cm)
        ylabel = True
        self._caxes = []
        if overlay:
            ax = self._axes[0]
            axtitle = None

        for i, layers in enumerate(epochs):
            stat = layers[0]
            if not overlay:
                ax = self._axes[i]

            # ax clusters
            if clusters_:
                if 'effect' in clusters_:
                    cs = clusters_.sub('effect == %r' % stat.name)
                else:
                    cs = clusters_
            else:
                cs = None

            cax = _ax_uts_clusters(ax, stat, cs, colors[i], pmax, ptrend, 'time',
                                   axtitle, ylabel)
            self._caxes.append(cax)
            ylabel = None

        self.clusters = clusters_
        self._show()

    def _fill_toolbar(self, tb):
        import wx

        btn = wx.Button(tb, wx.ID_ABOUT, "Clusters")
        tb.AddControl(btn)
        btn.Bind(wx.EVT_BUTTON, self._OnShowClusterInfo)

    def _OnShowClusterInfo(self, event):
        from .._wxutils import show_text_dialog
        info = str(self.clusters)
        show_text_dialog(self._frame, info, "Clusters")

    def set_pmax(self, pmax=0.05, ptrend=0.1):
        "set the threshold p-value for clusters to be displayed"
        for cax in self._caxes:
            cax.set_pmax(pmax, ptrend)
        self.draw()


def _ax_uts(ax, layers, title=False, bottom=None, top=None, invy=False,
            xlabel=True, ylabel=True, color=None, xdim='time'):
    contours = {}
    overlay = False
    for l in layers:
        args = _base.find_uts_args(l, overlay, color)
        overlay = True
        if args is None:
            continue

        _plt_uts(ax, l, xdim=xdim, **args)
        contours = l.info.get('contours', None)
        if contours:
            for v, color in contours.iteritems():
                if v in contours:
                    continue
                contours[v] = ax.axhline(v, color=color)

    l0 = layers[0]
    x = l0.get_dim(xdim)
    ax.set_xlim(x[0], x[-1])
    ax.x_fmt = "t = %.3f s"

    ticks = ax.xaxis.get_ticklocs()
    ticklabels = _base._ticklabels(ticks, 'time')
    ax.xaxis.set_ticklabels(ticklabels)

    if title:
        if 'name' in title:
            title = title.format(name=l0.name)
        ax.set_title(title)

    _base.set_xlabel(ax, xdim, xlabel)

    if ylabel is True:
        ylabel = l.info.get('unit', None)
    if ylabel:
        ax.set_ylabel(ylabel)

    if invy:
        y0, y1 = ax.get_ylim()
        bottom = bottom if (bottom is not None) else y1
        top = top if (top is not None) else y0
    if (bottom is not None) or (top is not None):
        ax.set_ylim(bottom, top)


def _plt_uts(ax, ndvar, color=None, xdim='time', kwargs={}):
    y = ndvar.get_data((xdim,))
    x = ndvar.get_dim(xdim).x
    if color is not None:
        kwargs['color'] = color
    ax.plot(x, y, **kwargs)

    for y, kwa in _base.find_uts_hlines(ndvar):
        if color is not None:
            kwa['color'] = color
        ax.axhline(y, **kwa)


class _ax_uts_clusters:
    def __init__(self, ax, Y, clusters, color=None, pmax=0.05, ptrend=0.1,
                 xdim='time', title=None, xlabel=True, ylabel=True):
        uts_args = _base.find_uts_args(Y, False, color)
        self._bottom, self._top = _base.find_vlim_args(Y)

        if title:
            if '{name}' in title:
                title = title.format(name=Y.name)
            ax.set_title(title)

        _plt_uts(ax, Y, xdim=xdim, **uts_args)

        if ylabel is True:
            ylabel = Y.info.get('meas', _base.default_meas)
        if ylabel:
            ax.set_ylabel(ylabel)

        xlabel = _base._axlabel(xdim, xlabel)
        if xlabel:
            ax.set_xlabel(xlabel)
        if np.any(Y.x < 0) and np.any(Y.x > 0):
            ax.axhline(0, color='k')


        # pmap
        self.cluster_plt = _plt_uts_clusters(ax, clusters, pmax, ptrend, color)

        # save ax attr
        self.ax = ax
        x = Y.get_dim(xdim).x
        self.xlim = (x[0], x[-1])

        ax.set_xlim(*self.xlim)
        ax.set_ylim(bottom=self._bottom, top=self._top)

    def set_clusters(self, clusters):
        self.cluster_plt.set_clusters(clusters)

    def set_pmax(self, pmax=0.05, ptrend=0.1):
        self.cluster_plt.set_pmax(pmax, ptrend)


class _plt_uts_clusters:
    def __init__(self, ax, clusters, pmax, ptrend, color=None, hatch='/'):
        """
        clusters : Dataset
            Dataset with entries for 'tstart', 'tstop' and 'p'.
        """
        self.pmax = pmax
        self.ptrend = ptrend
        self.h = []
        self.ax = ax
        self.clusters = clusters
        self.color = color
        self.hatch = hatch
        self.update()

    def set_clusters(self, clusters, update=True):
        self.clusters = clusters
        if update:
            self.update()

    def set_color(self, color, update=True):
        self.color = color
        if update:
            self.update()

    def set_pmax(self, pmax, ptrend, update=True):
        self.pmax = pmax
        self.ptrend = ptrend
        if update:
            self.update()

    def update(self):
        h = self.h
        while len(h):
            h.pop().remove()

        clusters = self.clusters
        if clusters is None:
            return

        p_include = self.ptrend or self.pmax
        for cluster in clusters.itercases():
            if 'p' in cluster:
                p = cluster['p']
                if p > p_include:
                    continue
                alpha = 0.5 if p < self.pmax else 0.2
            else:
                alpha = 0.5

            x0 = cluster['tstart']
            x1 = cluster['tstop']
            h = self.ax.axvspan(x0, x1, color=self.color,  # , hatch=self.hatch,
                                fill=True, alpha=alpha, zorder=-1)
            self.h.append(h)


class _plt_uts_stat(object):

    def __init__(self, ax, x, y, main, error, dev_data, color, label):
        # plot main
        if hasattr(main, '__call__'):
            y_main = main(y, axis=0)
            lw = mpl.rcParams['lines.linewidth']
            if error == 'all':
                lw *= 2
            self.main = ax.plot(x, y_main, color=color, label=label, lw=lw,
                                zorder=5)
        elif error == 'all':
            self.main = None
        else:
            raise ValueError("Invalid argument: main=%r" % main)

        # plot error
        if error == 'all':
            self.error = ax.plot(x, y.T, color=color, alpha=0.3)
        elif error:
            if error == 'data':
                pass
            elif hasattr(error, '__call__'):
                dev_data = error(y, axis=0)
            else:
                dev_data = stats.variability(y, None, None, error, False)
            lower = y_main - dev_data
            upper = y_main + dev_data
            self.error = ax.fill_between(x, lower, upper, color=color, alpha=0.3,
                                         linewidth=0, zorder=0)
        else:
            self.error = None
