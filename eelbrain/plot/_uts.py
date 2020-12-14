# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Plot uniform time-series of one variable."""
from typing import Any, Callable, Dict, Sequence, Tuple, Union

import matplotlib.axes
import matplotlib.colors
import matplotlib as mpl
import numpy as np

from .._colorspaces import oneway_colors, to_rgba
from .._data_obj import NDVarArg, CategorialArg, CellArg, IndexArg, Dataset, NDVar, cellname, longname
from . import _base
from ._base import (
    PlotType, EelFigure, PlotData, Layout,
    LegendMixin, YLimMixin, XAxisMixin, TimeSlicerEF,
    AxisData, StatLayer,
)
from ._styles import colors_for_oneway


class UTSStat(LegendMixin, XAxisMixin, YLimMixin, EelFigure):
    """
    Plot statistics for a one-dimensional NDVar

    Parameters
    ----------
    y : 1d-NDVar
        Dependent variable (one-dimensional NDVar).
    x : categorial
        Model: specification of conditions which should be plotted separately.
    xax : categorial
        Make separate axes for each category in this categorial model.
    match : categorial
        Identifier for repeated measures data.
    sub : index array
        Only use a subset of the data provided.
    ds
        If a Dataset is specified, all data-objects can be specified as
        names of Dataset variables.
    main : func | None
        Measure for the central tendency (function that takes an ``axis``
        argument). The default is numpy.mean.
    error
        Measure of variability to plot. Examples:
        ``sem``: Standard error of the mean;
        ``2sem``: 2 standard error of the mean;
        ``ci``: 95% confidence interval;
        ``99%ci``: 99% confidence interval.
        ``all``: Show all traces.
        ``none``: No variability indication.
    pool_error
        Pool the errors for the estimate of variability (default is True
        for related measures designs, False otherwise). See Loftus & Masson
        (1994).
    legend
        Matplotlib figure legend location argument or 'fig' to plot the
        legend in a separate figure.
    labels : dict
        Alternative labels for legend as ``{cell: label}`` dictionary (preserves
        order).
    axtitle
        Title for the individual axes. The default is to show the names of the
        epochs, but only if multiple axes are plotted.
    xlabel
        X-axis label. By default the label is inferred from the data.
    ylabel
        Y-axis label. By default the label is inferred from the data.
    xticklabels
        Specify which axes should be annotated with x-axis tick labels.
        Use ``int`` for a single axis, a sequence of ``int`` for multiple
        specific axes, or one of ``'left' | 'bottom' | 'all' | 'none'``.
    yticklabels
        Specify which axes should be annotated with y-axis tick labels.
        Use ``int`` for a single axis, a sequence of ``int`` for multiple
        specific axes, or one of ``'left' | 'bottom' | 'all' | 'none'``.
    invy
        Invert the y axis (if ``bottom`` and/or ``top`` are specified explicitly
        they take precedence; an inverted y-axis can also be produced by
        specifying ``bottom > top``).
    bottom
        The lower end of the plot's y axis.
    top
        The upper end of the plot's y axis.
    xdim
        dimension for the x-axis (default is 'time')
    xlim
        Initial x-axis view limits as ``(left, right)`` tuple or as ``length``
        scalar (default is the full x-axis in the data).
    clip
        Clip lines outside of axes (the default depends on whether ``frame`` is
        closed or open).
    color : matplotlib color
        Color if just a single category of data is plotted.
    colors
        Colors for the plots if multiple categories of data are plotted.
        **str**: A colormap name; Cells of ``x`` are mapped onto the colormap in
        regular intervals.
        **list**: A list of colors in the same sequence as ``x.cells``.
        **dict**: A dictionary mapping each cell in ``x`` to a color.
        Colors are specified as `matplotlib compatible color arguments
        <http://matplotlib.org/api/colors_api.html>`_.
    error_alpha
        Alpha of the error plot (default 0.3).
    mask : NDVar | {cell: NDVar}
        Mask certain time points.
    clusters
        Clusters to add to the plots. The clusters should be provided as
        Dataset, as stored in test results' :py:attr:`.clusters`.
    pmax
        Maximum p-value of clusters to plot as solid.
    ptrend
        Maximum p-value of clusters to plot as trend.
    tight
        Use matplotlib's tight_layout to expand all axes to fill the figure
        (default True)
    title
        Figure title.
    ...
        Also accepts :ref:`general-layout-parameters`.

    Notes
    -----
    Navigation:
     - ``↑``: scroll up
     - ``↓``: scroll down
     - ``r``: y-axis zoom in (reduce y-axis range)
     - ``c``: y-axis zoom out (increase y-axis range)
    """
    def __init__(
            self,
            y: NDVarArg,
            x: CategorialArg = None,
            xax: CategorialArg = None,
            match: CategorialArg = None,
            sub: IndexArg = None,
            ds: Dataset = None,
            main: Callable = np.mean,
            error: str = 'sem',
            pool_error: bool = None,
            legend: Union[str, int, bool] = 'upper right',
            labels: Dict[CellArg, str] = None,
            axtitle: Union[bool, Sequence[str]] = True,
            xlabel: Union[bool, str] = True,
            ylabel: Union[bool, str] = True,
            xticklabels: Union[str, int, Sequence[int]] = 'bottom',
            yticklabels: Union[str, int, Sequence[int]] = 'left',
            invy: bool = False,
            bottom: float = None,
            top: float = None,
            xdim: str = None,
            xlim: Union[float, Tuple[float, float]] = None,
            clip: bool = None,
            colors: Dict[CellArg, Any] = None,
            error_alpha: float = 0.3,
            mask: Union[NDVar, Dict[CellArg, NDVar]] = None,
            clusters: Dataset = None,
            pmax: float = 0.05,
            ptrend: float = 0.1,
            **kwargs):
        data = PlotData.from_stats(y, x, xax, match, sub, ds, (xdim,), colors, mask).for_plot(PlotType.LINE)
        xdim, = data.dims

        layout = Layout(data.plot_used, 2, 4, **kwargs)
        EelFigure.__init__(self, data.frame_title, layout)
        if clip is None:
            clip = layout.frame is True

        # create plots
        self._plots = []
        legend_handles = {}
        ymax = ymin = None
        for ax, ax_data in zip(self._axes, data):
            p = _ax_uts_stat(ax, ax_data, xdim, main, error, pool_error, clusters, pmax, ptrend, clip, error_alpha)
            self._plots.append(p)
            legend_handles.update(p.legend_handles)
            ymin = p.vmin if ymin is None else min(ymin, p.vmin)
            ymax = p.vmax if ymax is None else max(ymax, p.vmax)
        self._set_axtitle(axtitle, names=[ax_data.title for ax_data in data.plot_data])

        # axes limits
        if top is not None:
            ymax = top
        if bottom is not None:
            ymin = bottom
        if invy:
            ymin, ymax = ymax, ymin
        for p in self._plots:
            p.set_ylim(ymin, ymax)

        self._configure_axis_data('y', data.ct.y, ylabel, yticklabels)
        self._configure_axis_dim('x', data.ct.y.get_dim(xdim), xlabel, xticklabels)
        XAxisMixin._init_with_data(self, ((data.ct.y,),), xdim, xlim)
        YLimMixin.__init__(self, self._plots)
        LegendMixin.__init__(self, legend, legend_handles, labels)
        self._update_ui_cluster_button()
        self._show()

    def _fill_toolbar(self, tb):
        from .._wxgui import wx

        btn = self._cluster_btn = wx.Button(tb, wx.ID_ABOUT, "Clusters")
        btn.Enable(False)
        tb.AddControl(btn)
        btn.Bind(wx.EVT_BUTTON, self._OnShowClusterInfo)

        LegendMixin._fill_toolbar(self, tb)

    def _OnShowClusterInfo(self, event):
        from .._wxgui import show_text_dialog

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

    def set_clusters(
            self,
            clusters: Union[Dataset, None],
            pmax: float = 0.05,
            ptrend: float = None,
            color: Any = '.7',
            ax: int = None,
            y: Union[float, Dict] = None,
            dy: float = None,
    ):
        """Add clusters from a cluster test to the plot (as shaded area).

        Parameters
        ----------
        clusters
            The clusters, as stored in test results' :py:attr:`.clusters`.
            Use ``None`` to remove the clusters plotted on a given axis.
        pmax
            Only plot clusters with ``p <= pmax``.
        ptrend
            Maximum p-value of clusters to plot as trend.
        color : matplotlib color | dict
            Color for the clusters, or a ``{effect: color}`` dict.
        ax : None | int
            Index of the axes to which the clusters are to be added. If None,
            add the clusters to all axes.
        y
            Y level at which to plot clusters (default is boxes spanning the
            whole y-axis).
        dy
            Height of bars.
        """
        axes = range(len(self._axes)) if ax is None else [ax]

        # update plots
        for ax in axes:
            p = self._plots[ax].cluster_plt
            p.set_clusters(clusters, False)
            p.set_color(color, False)
            p.set_y(y, dy, False)
            p.set_pmax(pmax, ptrend)
        self.draw()

        self._update_ui_cluster_button()


class UTS(TimeSlicerEF, LegendMixin, YLimMixin, XAxisMixin, EelFigure):
    """Value by time plot for UTS data

    Parameters
    ----------
    y : (list of) NDVar
        Uts data to plot.
    xax : categorial
        Make separate axes for each category in this categorial model.
    axtitle : bool | sequence of str
        Title for the individual axes. The default is to show the names of the
        epochs, but only if multiple axes are plotted.
    ds : Dataset
        If a Dataset is specified, all data-objects can be specified as
        names of Dataset variables.
    sub : str | array
        Specify a subset of the data.
    xlabel
        X-axis label. By default the label is inferred from the data.
    ylabel
        Y-axis label. By default the label is inferred from the data.
    xticklabels
        Specify which axes should be annotated with x-axis tick labels.
        Use ``int`` for a single axis, a sequence of ``int`` for multiple
        specific axes, or one of ``'left' | 'bottom' | 'all' | 'none'``.
    yticklabels
        Specify which axes should be annotated with y-axis tick labels.
        Use ``int`` for a single axis, a sequence of ``int`` for multiple
        specific axes, or one of ``'left' | 'bottom' | 'all' | 'none'``.
    bottom
        The lower end of the plot's y axis.
    top
        The upper end of the plot's y axis.
    legend
        Matplotlib figure legend location argument or 'fig' to plot the
        legend in a separate figure.
    labels
        Alternative labels for legend as ``{cell: label}`` dictionary (preserves
        order).
    xlim
        Initial x-axis view limits as ``(left, right)`` tuple or as ``length``
        scalar (default is the full x-axis in the data).
    colors
        Dictionary mapping ``y`` names to color, or a single color to use for
        all lines.
    stem
        Plot as stem-plot (default is a line-plot).
    tight : bool
        Use matplotlib's tight_layout to expand all axes to fill the figure
        (default True)
    ...
        Also accepts :ref:`general-layout-parameters`.

    Notes
    -----
    Navigation:
     - ``↑``: scroll up
     - ``↓``: scroll down
     - ``←``: scroll left
     - ``→``: scroll right
     - ``home``: scroll to beginning
     - ``end``: scroll to end
     - ``f``: x-axis zoom in (reduce x axis range)
     - ``d``: x-axis zoom out (increase x axis range)
     - ``r``: y-axis zoom in (reduce y-axis range)
     - ``c``: y-axis zoom out (increase y-axis range)
    """
    def __init__(
            self,
            y: Union[NDVarArg, Sequence],
            xax: CategorialArg = None,
            axtitle: Union[bool, Sequence[str]] = True,
            ds: Dataset = None,
            sub: IndexArg = None,
            xlabel: Union[bool, str] = True,
            ylabel: Union[bool, str] = True,
            xticklabels: Union[str, int, Sequence[int]] = 'bottom',
            yticklabels: Union[str, int, Sequence[int]] = 'left',
            bottom: float = None,
            top: float = None,
            legend: Union[str, int, bool] = 'upper right',
            labels: Dict[CellArg, str] = None,
            xlim: Union[float, Tuple[float, float]] = None,
            colors: Union[Any, dict] = None,
            stem: bool = False,
            **kwargs):
        data = PlotData.from_args(y, (None,), xax, ds, sub)
        xdim = data.dims[0]
        layout = Layout(data.plot_used, 2, 4, **kwargs)
        EelFigure.__init__(self, data.frame_title, layout)
        self._set_axtitle(axtitle, data)
        self._configure_axis_dim('x', xdim, xlabel, xticklabels, data=data.data)
        self._configure_axis_data('y', data, ylabel, yticklabels)

        self.plots = []
        legend_handles = {}
        vlims = _base.find_fig_vlims(data.data, top, bottom)

        n_colors = max(map(len, data.data))
        if colors is None:
            colors_ = oneway_colors(n_colors)
        elif isinstance(colors, dict):
            colors_ = colors
        else:
            colors_ = (colors,) * n_colors

        for ax, layers in zip(self._axes, data.data):
            h = _ax_uts(ax, layers, xdim, vlims, colors_, stem)
            self.plots.append(h)
            legend_handles.update(h.legend_handles)

        self.epochs = data.data
        XAxisMixin._init_with_data(self, data.data, xdim, xlim)
        YLimMixin.__init__(self, self.plots)
        LegendMixin.__init__(self, legend, legend_handles, labels)
        TimeSlicerEF.__init__(self, xdim, data.time_dim)
        self._show()

    def _fill_toolbar(self, tb):
        LegendMixin._fill_toolbar(self, tb)


class _ax_uts_stat:

    def __init__(
            self,
            ax: matplotlib.axes.Axes,
            data: AxisData,
            xdim: str,
            main: Callable,
            error: Union[str, Callable],
            pool_error: bool,
            clusters,
            pmax,
            ptrend,
            clip: bool,
            error_alpha,
    ):
        # stat plots
        self.ax = ax
        self.stat_plots = []
        self.legend_handles = {}

        for layer in data:
            plt = _plt_uts_stat(ax, layer, xdim, main, error, pool_error, clip, error_alpha)
            self.stat_plots.append(plt)
            if plt.main is not None:
                self.legend_handles[layer.cell] = plt.main[0]

        # cluster plot
        self.cluster_plt = _plt_uts_clusters(ax, clusters, pmax, ptrend)

        # format y axis
        ax.autoscale(True, 'y')
        ax.autoscale(False, 'x')
        self.vmin, self.vmax = self.ax.get_ylim()

    @property
    def title(self):
        return self.ax.get_title()

    def set_ylim(self, vmin, vmax):
        self.ax.set_ylim(vmin, vmax)
        self.vmin, self.vmax = self.ax.get_ylim()


class UTSClusters(EelFigure):
    """Plot permutation cluster test results

    Parameters
    ----------
    res : testnd.ANOVA
        ANOVA with permutation cluster test result object.
    pmax : scalar
        Maximum p-value of clusters to plot as solid.
    ptrend : scalar
        Maximum p-value of clusters to plot as trend.
    axtitle : bool | sequence of str
        Title for the individual axes. The default is to show the names of the
        epochs, but only if multiple axes are plotted.
    cm : str
        Colormap to use for coloring different effects.
    overlay : bool
        Plot epochs (time course for different effects) on top of each
        other (as opposed to on separate axes).
    xticklabels
        Specify which axes should be annotated with x-axis tick labels.
        Use ``int`` for a single axis, a sequence of ``int`` for multiple
        specific axes, or one of ``'left' | 'bottom' | 'all' | 'none'``.
    yticklabels
        Specify which axes should be annotated with y-axis tick labels.
        Use ``int`` for a single axis, a sequence of ``int`` for multiple
        specific axes, or one of ``'left' | 'bottom' | 'all' | 'none'``.
    tight : bool
        Use matplotlib's tight_layout to expand all axes to fill the figure
        (default True)
    ...
        Also accepts :ref:`general-layout-parameters`.
    """
    def __init__(
            self, res, pmax=0.05, ptrend=0.1, axtitle=True, cm=None,
            overlay=False,
            xticklabels: Union[str, int, Sequence[int]] = 'bottom',
            yticklabels: Union[str, int, Sequence[int]] = 'left',
            **kwargs):
        clusters_ = res.clusters

        data = PlotData.from_args(res, (None,))
        xdim = data.dims[0]
        # create figure
        layout = Layout(1 if overlay else data.plot_used, 2, 4, **kwargs)
        EelFigure.__init__(self, data.frame_title, layout)
        self._set_axtitle(axtitle, data)

        colors = colors_for_oneway(range(data.n_plots), cmap=cm)
        self._caxes = []
        if overlay:
            ax = self._axes[0]

        for i, layers in enumerate(data.data):
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

            cax = _ax_uts_clusters(ax, stat, cs, colors[i], pmax, ptrend, xdim)
            self._caxes.append(cax)

        self._configure_axis_data('y', data, True, yticklabels)
        self._configure_axis_dim('x', xdim, True, xticklabels, data=data.data)
        self.clusters = clusters_
        self._show()

    def _fill_toolbar(self, tb):
        from .._wxgui import wx

        btn = wx.Button(tb, wx.ID_ABOUT, "Clusters")
        tb.AddControl(btn)
        btn.Bind(wx.EVT_BUTTON, self._OnShowClusterInfo)

    def _OnShowClusterInfo(self, event):
        from .._wxgui import show_text_dialog

        info = str(self.clusters)
        show_text_dialog(self._frame, info, "Clusters")

    def set_pmax(self, pmax=0.05, ptrend=0.1):
        "Set the threshold p-value for clusters to be displayed"
        for cax in self._caxes:
            cax.set_pmax(pmax, ptrend)
        self.draw()


class _ax_uts:

    def __init__(self, ax, layers, xdim, vlims, colors, stem):
        vmin, vmax = _base.find_uts_ax_vlim(layers, vlims)
        if isinstance(colors, dict):
            colors = [colors[l.name] for l in layers]

        self.legend_handles = {}
        for l, color in zip(layers, colors):
            color = l.info.get('color', color)
            p = _plt_uts(ax, l, xdim, color, stem)
            self.legend_handles[longname(l)] = p.plot_handle
            contours = l.info.get('contours', None)
            if contours:
                for v, c in contours.items():
                    if v in contours:
                        continue
                    contours[v] = ax.axhline(v, color=c)

        self.ax = ax
        self.set_ylim(vmin, vmax)

    def set_ylim(self, vmin, vmax):
        self.ax.set_ylim(vmin, vmax)
        self.vmin, self.vmax = self.ax.get_ylim()


class _plt_uts:

    def __init__(self, ax, ndvar, xdim, color, stem=False):
        y = ndvar.get_data((xdim,))
        x = ndvar.get_dim(xdim)._axis_data()
        label = longname(ndvar)
        if stem:
            nonzero = y != 0
            nonzero[0] = True
            nonzero[-1] = True
            color = matplotlib.colors.to_hex(color)
            self.plot_handle = ax.stem(x[nonzero], y[nonzero], bottom=0, linefmt=color, markerfmt=' ', basefmt=f'#808080', use_line_collection=True, label=label)
        else:
            self.plot_handle = ax.plot(x, y, color=color, label=label)[0]

        for y, kwa in _base.find_uts_hlines(ndvar):
            if color is not None:
                kwa['color'] = color
            ax.axhline(y, **kwa)


class _ax_uts_clusters:
    def __init__(self, ax, y, clusters, color=None, pmax=0.05, ptrend=0.1,
                 xdim='time'):
        self._bottom, self._top = _base.find_vlim_args(y)
        if color is None:
            color = y.info.get('color')

        _plt_uts(ax, y, xdim, color)

        if np.any(y.x < 0) and np.any(y.x > 0):
            ax.axhline(0, color='k')

        # pmap
        self.cluster_plt = _plt_uts_clusters(ax, clusters, pmax, ptrend, color)

        # save ax attr
        self.ax = ax
        x = y.get_dim(xdim)._axis_data()
        self.xlim = (x[0], x[-1])

        ax.set_xlim(*self.xlim)
        ax.set_ylim(bottom=self._bottom, top=self._top)

    def set_clusters(self, clusters):
        self.cluster_plt.set_clusters(clusters)

    def set_pmax(self, pmax=0.05, ptrend=0.1):
        self.cluster_plt.set_pmax(pmax, ptrend)


class _plt_uts_clusters:
    """UTS cluster plot"""
    def __init__(
            self,
            ax: matplotlib.axes.Axes,
            clusters: Dataset,  # 'tstart', 'tstop', 'p', 'effect'
            pmax: float,
            ptrend: float,
            color: Any = None,
    ):
        self.pmax = pmax
        self.ptrend = ptrend
        self.h = []
        self.ax = ax
        self.clusters = clusters
        self.color = color
        self.y = None
        self.dy = None
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

    def set_y(self, y, dy, update=True):
        self.y = y
        self.dy = dy
        if update:
            self.update()

    def update(self):
        h = self.h
        while len(h):
            h.pop().remove()

        clusters = self.clusters
        if clusters is None:
            return

        if self.dy is None:
            bottom, top = self.ax.get_ylim()
            dy = (top - bottom) / 40.
        else:
            dy = self.dy

        p_include = self.ptrend or self.pmax
        for cluster in clusters.itercases():
            if 'p' in cluster:
                p = cluster['p']
                if p > p_include:
                    continue
                alpha = 0.5 if p < self.pmax else 0.2
            else:
                alpha = 0.5

            tstart = cluster['tstart']
            tstop = cluster['tstop']
            effect = cluster.get('effect')
            color = self.color[effect] if isinstance(self.color, dict) else self.color
            y = self.y[effect] if isinstance(self.y, dict) else self.y
            if y is None:
                h = self.ax.axvspan(
                    tstart, tstop, color=color, fill=True, alpha=alpha, zorder=-10)
            else:
                h = mpl.patches.Rectangle(
                    (tstart, y - dy / 2.), tstop - tstart, dy, facecolor=color,
                    linewidth=0, zorder=-10)
                self.ax.add_patch(h)
            self.h.append(h)


class _plt_uts_stat:

    def __init__(
            self,
            ax: matplotlib.axes.Axes,
            layer: StatLayer,
            xdim: str,
            main: Callable,
            error: Union[str, Callable],
            pool_error: bool,
            clip: bool,
            error_alpha: float,
    ):
        # zorder defaults: 1 for patches, 2 for lines
        label = cellname(layer.cell)
        x = layer.y.get_dim(xdim)._axis_data()
        # plot main
        if callable(main):
            y_main = layer.get_statistic(main)
            kwargs = layer.style.line_args
            if error == 'all':
                lw = kwargs['linewidth'] or mpl.rcParams['lines.linewidth']
                kwargs = {**kwargs, 'linewidth': lw * 2}
            self.main = ax.plot(x, y_main, label=label, clip_on=clip, **kwargs)
        elif error == 'all':
            self.main = y_main = None
        else:
            raise TypeError(f"main={main!r}")

        # plot error
        if error == 'all':
            y_all = layer.y.get_data((xdim, 'case'))
            self.error = ax.plot(x, y_all, alpha=error_alpha, clip_on=clip, **layer.style.line_args)
        elif error and error != 'none':
            if callable(error):
                dev_data = layer.get_statistic(error)
            else:
                dev_data = layer.get_dispersion(error, pool_error)
            lower = y_main - dev_data
            upper = y_main + dev_data
            r, g, b, a = to_rgba(layer.style.color)
            a *= error_alpha
            self.error = ax.fill_between(x, lower, upper, color=(r, g, b, a), linewidth=0, zorder=1.99, clip_on=clip)
        else:
            self.error = None
