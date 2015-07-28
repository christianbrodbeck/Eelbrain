"""
Plot multidimensional uniform time series.
"""

from __future__ import division

from itertools import izip
import math

import numpy as np

from .. import _data_obj as _dta
from .._names import INTERPOLATE_CHANNELS
from . import _base
from ._base import _EelFigure


class _plt_im_array(object):
    def __init__(self, ax, ndvar, overlay, dims=('time', 'sensor'),
                 extent=None, interpolation=None, vlims={}, cmaps={},
                 contours={}):
        im_kwa = _base.find_im_args(ndvar, overlay, vlims, cmaps)
        self._contours = _base.find_ct_args(ndvar, overlay, contours)
        self._meas = ndvar.info.get('meas', _base.default_meas)
        self._dims = dims

        data = self._data_from_ndvar(ndvar)
        if im_kwa is not None:
            self.im = ax.imshow(data, origin='lower', aspect='auto',
                                extent=extent, interpolation=interpolation,
                                **im_kwa)
            self._cmap = im_kwa['cmap']
        else:
            self.im = None

        # store attributes
        self.ax = ax
        self.cont = None
        self._data = data
        self._aspect = 'auto'
        self._extent = extent

        # draw flexible part
        self._draw_contours()

    def _data_from_ndvar(self, ndvar):
        data = ndvar.get_data(self._dims)
        if data.ndim > 2:
            assert data.shape[0] == 1
            data = data[0]
        return data

    def _draw_contours(self):
        if self.cont:
            for c in self.cont.collections:
                c.remove()
            self.cont = None

        if self._contours:
            levels = sorted(self._contours)
            colors = [self._contours[l] for l in levels]
            self.cont = self.ax.contour(self._data, levels=levels,
                                        colors=colors, aspect=self._aspect,
                                        origin='lower', extent=self._extent)

    def add_contour(self, meas, level, color):
        if self._meas == meas:
            self._contours[level] = color
            self._draw_contours()

    def get_kwargs(self):
        "Get the arguments required for plotting the im"
        if self.im:
            vmin, vmax = self.im.get_clim()
            args = dict(cmap=self._cmap, vmin=vmin, vmax=vmax)
        else:
            args = {}
        return args

    def set_cmap(self, cmap, meas=None):
        if (self.im is not None) and (meas is None or meas == self._meas):
            self.im.set_cmap(cmap)
            self._cmap = cmap

    def set_data(self, ndvar):
        data = self._data_from_ndvar(ndvar)
        if self.im is not None:
            self.im.set_data(data)
        self._data = data
        self._draw_contours()

    def set_vlim(self, vmax, meas, vmin):
        if self.im is None:
            return
        elif (meas is not None) and (self._meas != meas):
            return

        if vmax is None:
            _, vmax = self.im.get_clim()

        vmin, vmax = _base.fix_vlim_for_cmap(vmin, vmax, self._cmap)
        self.im.set_clim(vmin, vmax)


class _ax_im_array(object):
    def __init__(self, ax, layers, x='time', title=None, interpolation=None,
                 vlims={}, cmaps={}, contours={}):
        """
        plots segment data as im

        define a colorspace by supplying one of those kwargs: ``colorspace`` OR
        ``p`` OR ``vmax``

        """
        self.ax = ax
        self.data = layers
        self.layers = []
        epoch = layers[0]

        xdim = epoch.get_dim(x)
        if epoch.ndim == 2:
            xdim_i = epoch.dimnames.index(x)
            ydim_i = {1:0, 0:1}[xdim_i]
            y = epoch.dimnames[ydim_i]
        else:
            err = ("Need 2 dimensions, got %i" % epoch.ndim)
            raise ValueError(err)

        ydim = epoch.get_dim(y)
        if not ydim.adjacent:
            ydim = _dta.Var(np.arange(len(ydim)), y)

        # plot
        overlay = False
        extent = (xdim[0], xdim[-1], ydim[0], ydim[-1])
        for l in layers:
            p = _plt_im_array(ax, l, overlay, (y, x), extent, interpolation,
                              vlims, cmaps, contours)
            self.layers.append(p)
            overlay = True

        # title
        if title is None:
            title = _base.str2tex(epoch.name)
        ax.set_title(title)

    def add_contour(self, meas, level, color):
        for l in self.layers:
            l.add_contour(meas, level, color)

    def set_cmap(self, cmap, meas=None):
        """Change the colormap in the array plot

        Parameters
        ----------
        cmap : str | colormap
            New colormap.
        meas : None | str
            Measurement to which to apply the colormap. With None, it is
            applied to all.
        """
        for l in self.layers:
            l.set_cmap(cmap, meas)

    def set_data(self, layers):
        self.data = layers
        for l, p in zip(layers, self.layers):
            p.set_data(l)

    def set_vlim(self, vmax=None, meas=None, vmin=None):
        for l in self.layers:
            l.set_vlim(vmax, meas, vmin)


class Array(_EelFigure):
    """Plot UTS data to a rectangular grid.

    Parameters
    ----------
    epochs : NDVar
        If data has only 1 dimension, the x-axis defines epochs.
    Xax : None | categorial
        Create a separate plot for each cell in this model.
    xlabel, ylabel : bool | str
        Labels for x- and y-axis; the default is determined from the data.
    xticklabels : bool
        Add tick-labels to the x-axis (default True).
    ds : None | Dataset
        If a Dataset is provided, ``epochs`` and ``Xax`` can be specified
        as strings.
    x : str
        Dimension to plot on the x axis (default 'time').
    vmax : scalar
        Upper limits for the colormap.
    vmin : scalar
        Lower limit for the colormap.
    tight : bool
        Use matplotlib's tight_layout to expand all axes to fill the figure
        (default True)
    title : None | string
        Figure title.
    """
    def __init__(self, epochs, Xax=None, xlabel=True, ylabel=True,
                 xticklabels=True, ds=None, x='time', vmax=None, vmin=None,
                 *args, **kwargs):
        epochs, (xdim, ydim) = _base.unpack_epochs_arg(epochs, (x, None), Xax, ds)

        nax = len(epochs)
        _EelFigure.__init__(self, "Array Plot", nax, 4, 2, *args, **kwargs)

        self.plots = []
        vlims = _base.find_fig_vlims(epochs, False, vmax, vmin)
        for i, ax, layers in zip(xrange(nax), self._axes, epochs):
            p = _ax_im_array(ax, layers, x, vlims=vlims)
            self.plots.append(p)

        self._configure_xaxis_dim(xdim, xlabel, xticklabels)
        self._configure_yaxis_dim(ydim, ylabel)
        self._show()

    def add_contour(self, meas, level, color='k'):
        """Add a contour line

        Parameters
        ----------
        meas : str
            The measurement for which to add a contour line.
        level : scalar
            The value at which to draw the contour.
        color : matplotlib color
            The color of the contour line.
        """
        for p in self.plots:
            p.add_contour(meas, level, color)
        self.draw()

    def set_cmap(self, cmap, meas=None):
        """Change the colormap in the array plots

        Parameters
        ----------
        cmap : str | colormap
            New colormap.
        meas : None | str
            Measurement to which to apply the colormap. With None, it is
            applied to all.
        """
        for p in self.plots:
            p.set_cmap(cmap, meas)
        self.draw()

    def set_vlim(self, vmax=None, meas=None, vmin=None):
        for p in self.plots:
            p.set_vlim(vmax, meas, vmin)
        self.draw()


class _plt_utsnd:

    def __init__(self, ax, epoch, sensors=None, *args, **kwargs):
        """
        uts plot for a single epoch

        Parameters
        ----------
        ax : matplotlib axes
            Target axes.
        epoch : NDVar (sensor by time)
            Epoch to plot.
        sensors : None | True | numpy index
            The sensors to plot (None or True -> all sensors).
        others :
            Matplotlib plot() arguments.
        """
        if sensors is not None and sensors is not True:
            epoch = epoch.sub(sensor=sensors)

        kwargs['label'] = epoch.name
        self.lines = ax.plot(epoch.time.x, epoch.get_data(('time', 'sensor')),
                             *args, **kwargs)

        for y, kwa in _base.find_uts_hlines(epoch):
            ax.axhline(y, **kwa)

        self.epoch = epoch
        self._sensors = sensors

    def remove(self):
        while self.lines:
            self.lines.pop().remove()

    def set_ydata(self, epoch):
        if self._sensors:
            epoch = epoch.sub(sensor=self._sensors)
        for line, y in izip(self.lines, epoch.get_data(('sensor', 'time'))):
            line.set_ydata(y)


class _plt_extrema:

    def __init__(self, ax, epoch, **plot_kwargs):
        self.upper = ax.plot(epoch.time.x, epoch.max('sensor').x, **plot_kwargs)[0]
        self.lower = ax.plot(epoch.time.x, epoch.min('sensor').x, **plot_kwargs)[0]
        self.epoch = epoch

    def remove(self):
        self.upper.remove()
        self.lower.remove()

    def set_ydata(self, epoch):
        self.upper.set_ydata(epoch.max('sensor').x)
        self.lower.set_ydata(epoch.min('sensor').x)


class _ax_butterfly(object):

    def __init__(self, ax, layers, sensors=None, extrema=False, title='{name}',
                 color=None, vlims={}):
        """
        Parameters
        ----------
        vmin, vmax: None | scalar
            Y axis limits.
        """
        self.ax = ax
        self.data = layers
        self.layers = []
        self._xvalues = []  # values on the x axis
        self._meas = None

        vmin, vmax = _base.find_uts_ax_vlim(layers, vlims)

        name = ''
        overlay = False
        for l in layers:
            uts_args = _base.find_uts_args(l, overlay, color)
            if uts_args is None:
                continue
            overlay = True

            # plot
            if extrema:
                h = _plt_extrema(ax, l, **uts_args)
            else:
                h = _plt_utsnd(ax, l, sensors, **uts_args)

            self.layers.append(h)
            if not name:
                name = getattr(l, 'name', '')

            self._xvalues = np.union1d(self._xvalues, l.time.x)

        # axes decoration
        ax.set_xlim(min(l.epoch.time[0] for l in self.layers),
                    max(l.epoch.time[-1] for l in self.layers))

    #    ax.yaxis.set_offset_position('right')
        ax.yaxis.offsetText.set_va('top')

        if isinstance(title, str):
            ax.set_title(title.format(name=name))

        self.set_vlim(vmax, vmin)

    def set_vlim(self, vmax=None, vmin=None):
        if vmin is None and vmax is not None:
            vmin = -vmax
        self.ax.set_ylim(vmin, vmax)
        vmin, vmax = self.ax.get_ylim()
        self.vmin = vmin
        self.vmax = vmax


class Butterfly(_EelFigure):
    """Butterfly plot for NDVars

    Parameters
    ----------
    epochs : (list of) NDVar
        Data to plot.
    Xax : None | categorial
        Create a separate plot for each cell in this model.
    sensors: None or list of sensor IDs
        sensors to plot (``None`` = all)
    axtitle : str | None
        Title to plot for axes. Default is the NDVar names.
    xlabel : str | None
        X-axis labels. By default the label is inferred from the data.
    ylabel : str | None
        Y-axis labels. By default the label is inferred from the data.
    xticklabels : bool
        Add tick-labels to the x-axis (default True).
    color : matplotlib color
        default (``None``): use segment color if available, otherwise
        black; ``True``: alternate colors (mpl default)
    ds : None | Dataset
        If a Dataset is provided, ``epochs`` and ``Xax`` can be specified
        as strings.
    tight : bool
        Use matplotlib's tight_layout to expand all axes to fill the figure
        (default True)
    title : None | string
        Figure title.
    """
    def __init__(self, epochs, Xax=None, sensors=None, axtitle='{name}',
                 xlabel=True, ylabel=True, xticklabels=True, color=None,
                 ds=None, *args, **kwargs):
        epochs, (xdim, _) = _base.unpack_epochs_arg(epochs, ('time', None), Xax, ds)
        _EelFigure.__init__(self, 'Butterfly Plot', len(epochs), 4, 2, *args,
                            **kwargs)
        self._configure_xaxis_dim(xdim, xlabel, xticklabels)
        self._configure_yaxis(epochs[0][0], ylabel)

        self.plots = []
        vlims = _base.find_fig_vlims(epochs, True)
        for ax, layers in zip(self._axes, epochs):
            h = _ax_butterfly(ax, layers, sensors, False, axtitle, color, vlims)
            self.plots.append(h)

        self._show()

    def set_vlim(self, vmax=None, vmin=None):
        for p in self.plots:
            p.set_vlim(vmax, vmin)
        self.draw()


class _ax_bfly_epoch:
    def __init__(self, ax, epoch, mark=None, state=True, color='k', lw=0.2,
                 mcolor='r', mlw=0.8, antialiased=True, vlims={},
                 plot_range=False):
        """Specific plot for showing a single sensor by time epoch

        Parameters
        ----------
        ax : mpl Axes
            Plot target axes.
        epoch : NDVar
            Sensor by time epoch.
        mark : dict {int: mpl color}
            Channel: color dict of channels with custom color.
        color : mpl color
            Color for unmarked traces.
        lw : scalar
            Sensor trace plot Line width (default 0.5).
        mlw : scalar
            Marked sensor plot line width (default 1).
        """
        if plot_range:
            self.extrema = _plt_extrema(ax, epoch, color=color,
                                        antialiased=antialiased)
            if mark:
                self.lines = _plt_utsnd(ax, epoch, sorted(mark), color='r',
                                        lw=mlw, antialiased=antialiased)
                mark = None
            else:
                self.lines = None
        else:
            self.extrema = None
            self.lines = _plt_utsnd(ax, epoch, color=color, lw=lw,
                                    antialiased=antialiased)

        self.ax = ax
        self.epoch = epoch
        self._state_h = []
        self._ylim = _base.find_uts_ax_vlim([epoch], vlims)
        self._set_ax_lim()
        self._styles = {None: {'color': color, 'lw': lw, 'ls': '-',
                               'zorder': 2},
                        'mark': {'color': mcolor, 'lw': mlw, 'ls': '-',
                                 'zorder': 10},
                        INTERPOLATE_CHANNELS: {'color': 'b', 'lw': 1.2,
                                               'ls': ':', 'zorder': 6}}
        self._marked = {'mark': set(), INTERPOLATE_CHANNELS: set()}
        if mark:
            self.set_marked('mark', mark)

        # create initial plots
        self.set_state(state)

    def set_data(self, epoch):
        if self.extrema is not None:
            self.extrema.set_ydata(epoch)
        if self.lines is not None:
            self.lines.set_ydata(epoch)

    def _set_ax_lim(self):
        self.ax.set_xlim(self.epoch.time[0], self.epoch.time[-1])
        ylim = self._ylim
        if ylim:
            if np.isscalar(ylim):
                self.ax.set_ylim(-ylim, ylim)
            else:
                y_min, y_max = ylim
                self.ax.set_ylim(y_min, y_max)

    def set_marked(self, kind, sensors):
        """Set the channels which should be marked for a specific style

        Parameters
        ----------
        kind : str
            The style.
        sensors : collection of int
            Channel index for the channels to mark as ``kind``.
        """
        old = self._marked[kind]
        new = self._marked[kind] = set(sensors)
        if not old and not new:
            return
        elif self.extrema is not None:
            raise NotImplementedError("Set styles with extrema plot")
        # mark new channels
        for i in new.difference(old):
            self.lines.lines[i].update(self._styles[kind])
        # find channels to unmark
        old.difference_update(new)
        if not old:
            return
        # possible alternate style
        if kind == 'mark':
            other_kind = INTERPOLATE_CHANNELS
        else:
            other_kind = 'mark'
        # reset old channels
        for i in old:
            if i in self._marked[other_kind]:
                self.lines.lines[i].update(self._styles[other_kind])
            else:
                self.lines.lines[i].update(self._styles[None])


    def set_state(self, state):
        "Set the state (True=accept / False=reject)"
        if state:
            while self._state_h:
                h = self._state_h.pop()
                h.remove()
        else:
            if not self._state_h:
                h1 = self.ax.plot([0, 1], [0, 1], color='r', linewidth=1,
                                  transform=self.ax.transAxes)
                h2 = self.ax.plot([0, 1], [1, 0], color='r', linewidth=1,
                                  transform=self.ax.transAxes)
                self._state_h.extend(h1 + h2)

    def set_ylim(self, ylim):
        self._ylim = ylim
        self._set_ax_lim()
