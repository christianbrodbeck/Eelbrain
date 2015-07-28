"""
Plot topographic maps of sensor space data.
"""
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from __future__ import division

from itertools import izip
from math import floor, sqrt

import numpy as np
from scipy import interpolate

from . import _base
from ._base import _EelFigure
from . import _utsnd as _utsnd
from ._sensors import SensorMapMixin
from ._sensors import _plt_map2d



class Topomap(SensorMapMixin, _EelFigure):
    """Plot individual topogeraphies

    Parameters
    ----------
    epochs : NDVar | list of NDVar, dims = ([case,] sensor,)
        Data to plot.
    Xax : None | categorial
        Create a separate plot for each cell in this model.
    sensorlabels : None | 'index' | 'name' | 'fullname'
        Show sensor labels. For 'name', any prefix common to all names
        is removed; with 'fullname', the full name is shown.
    proj : str
        The sensor projection to use for topomaps.
    res : int
        Resolution of the topomaps (width = height = ``res``).
    interpolation : str
        Matplotlib imshow() parameter for topomaps.
    ds : None | Dataset
        If a Dataset is provided, ``epochs`` and ``Xax`` can be specified
        as strings.
    vmax, vmin : None | scalar
        Override the default plot limits. If only vmax is specified, vmin
        is set to -vmax.
    axtitle : str | None
        Axes title, default is each topography's name.
    xlabel : str
        Label below the topomaps (default is no label).
    tight : bool
        Use matplotlib's tight_layout to expand all axes to fill the figure
        (default True)
    title : None | string
        Figure title.
    """
    def __init__(self, epochs, Xax=None, sensorlabels='name', proj='default',
                 res=200, interpolation='nearest', ds=None, vmax=None,
                 vmin=None, axtitle=True, xlabel=None, *args, **kwargs):
        epochs, _ = self._epochs = _base.unpack_epochs_arg(epochs, ('sensor',), Xax, ds)
        nax = len(epochs)

        _EelFigure.__init__(self, "Topomap", nax, 7, 1, *args, **kwargs)

        vlims = _base.find_fig_vlims(epochs, True, vmax, vmin)

        self._plots = []
        sensor_plots = []
        for i, ax, layers in izip(xrange(nax), self._axes, epochs):
            ax.ID = i
            h = _ax_topomap(ax, layers, axtitle, sensorlabels, res=res, proj=proj,
                            interpolation=interpolation, xlabel=xlabel,
                            vlims=vlims)
            self._plots.append(h)
            sensor_plots.append(h.sensors)

        SensorMapMixin.__init__(self, sensor_plots, sensorlabels)
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
        for p in self._plots:
            p.add_contour(meas, level, color)
        self.draw()

    def set_cmap(self, cmap, base=True, overlays=False, **kwa):
        """Change the colormap in the topomaps

        Parameters
        ----------
        cmap : str | colormap
            New colormap.
        base : bool
            Apply the new colormap in the lowest layer of each plot.
        overlays : bool
            Apply the new colormap to the layers above the first layer.
        """
        for p in self._plots:
            p.set_cmap(cmap, base, overlays)
        self.draw()

    def set_vlim(self, vmax=None, meas=None, vmin=None):
        "Change the range of the data shown in he colormap"
        for p in self._plots:
            p.set_vlim(vmax, meas, vmin)
        self.draw()


class TopomapBins(_EelFigure):

    def __init__(self, epochs, Xax=None, bin_length=0.05, tstart=None,
                 tstop=None, ds=None, vmax=None, vmin=None, *args, **kwargs):
        epochs, _ = _base.unpack_epochs_arg(epochs, ('sensor', 'time'), Xax, ds)
        epochs = [[l.bin(bin_length, tstart, tstop) for l in layers]
                  for layers in epochs]

        # create figure
        time = epochs[0][0].get_dim('time')
        n_bins = len(time)
        n_rows = len(epochs)
        _EelFigure.__init__(self, "TopomapBins Plot", n_bins * n_rows, 1.5, 1,
                            False, *args, nrow=n_rows, ncol=n_bins, **kwargs)

        vlims = _base.find_fig_vlims(epochs, True, vmax, vmin)

        for row, layers in enumerate(epochs):
            for column, t in enumerate(time.x):
                if row == 0:
                    title = str(t)
                else:
                    title = None
                ax = self._axes[row * n_bins + column]
                topo_layers = [l.sub(time=t) for l in layers]
                _ax_topomap(ax, topo_layers, title, vlims=vlims)

        self._show()


class TopoButterfly(_EelFigure):
    """Butterfly plot with corresponding topomaps

    Parameters
    ----------
    epochs :
        Epoch(s) to plot.
    Xax : None | categorial
        Create a separate plot for each cell in this model.
    xlabel, ylabel : bool | string
        Labels for x and y axes. If True, labels are automatically chosen.
    xticklabels : bool
        Add tick-labels to the x-axis (default True).
    proj : str
        The sensor projection to use for topomaps.
    res : int
        Resolution of the topomaps (width = height = ``res``).
    interpolation : str
        Matplotlib imshow() parameter for topomaps.
    color : matplotlib color
        Color of the butterfly plots.
    sensorlabels : None | 'index' | 'name' | 'fullname'
        Show sensor labels. For 'name', any prefix common to all names
        is removed; with 'fullname', the full name is shown.
    mark : None | list of sensor names or indices
        Highlight a subset of the sensors.
    mcolor : matplotlib color
        Color for marked sensors.
    ds : None | Dataset
        If a Dataset is provided, ``epochs`` and ``Xax`` can be specified
        as strings.
    axh : scalar
        Height of the butterfly axes as well as side length of the topomap
        axes (in inches).
    ax_aspect : scalar
        multiplier for the width of butterfly plots based on their height
    vmax, vmin : None | scalar
        Override the default plot limits. If only vmax is specified, vmin
        is set to -vmax.
    title : None | string
        Figure title.

    Notes
    -----
     - LMB click in butterfly plots fixates the topomap time.
     - RMB click in butterfly plots removes the time point, the topomaps follow
       the mouse pointer.
     - ``Right arrow``: Increment the current topomap time.
     - ``Left arrow``: Decrement the current topomap time.
     - ``t``: open a TopoMap plot for the region under the mouse pointer.
    """
    _default_xlabel_ax = -2
    _make_axes = False

    def __init__(self, epochs, Xax=None, xlabel=True, ylabel=True,
                 xticklabels=True,
                 proj='default', res=100, interpolation='nearest', color=None,
                 sensorlabels=None, mark=None, mcolor=None, ds=None, vmax=None,
                 vmin=None, *args, **kwargs):
        epochs, _ = _base.unpack_epochs_arg(epochs, ('sensor', 'time'), Xax, ds)
        n_plots = len(epochs)
        self._epochs = epochs

        # create figure
        nax = 3 * n_plots  # for layout pretend butterfly & topo are 3 axes
        kwargs['ncol'] = 3
        _EelFigure.__init__(self, "TopoButterfly Plot", nax, 3, 1, False, *args,
                            **kwargs)

        # axes sizes
        frame = .05  # in inches; .4
        w = self._layout.w
        h = self._layout.h
        ax_aspect = 2

        xframe = frame / w
        x_left_ylabel = 0.5 / w if ylabel else 0
        x_left_title = 0.5 / w
        x_text = x_left_title / 3
        ax1_left = xframe + x_left_title + x_left_ylabel
        ax1_width = ax_aspect / (ax_aspect + 1) - ax1_left - xframe / 2
        ax2_left = ax_aspect / (ax_aspect + 1) + xframe / 2
        ax2_width = 1 / (ax_aspect + 1) - 1.5 * xframe

        yframe = frame / h
        y_bottomframe = 0.5 / h
        y_sep = (1 - y_bottomframe) / n_plots
        height = y_sep - yframe

        self._topo_kwargs = {'proj': proj,
                             'res': res,
                             'interpolation': interpolation,
                             'sensorlabels': sensorlabels,
                             'mark': mark,
                             'mcolor': mcolor,
                             'title': False}

        vlims = _base.find_fig_vlims(epochs, True, vmax, vmin)

        self.bfly_axes = []
        self.topo_axes = []
        self.bfly_plots = []
        self.topo_plots = []
        self._topoax_data = []
        self.t_markers = []
        self._vlims = vlims
        self._xvalues = []

        # plot epochs (x/y are in figure coordinates)
        for i, layers in enumerate(epochs):
            # position axes
            bottom = 1 - y_sep * (1 + i)

            ax1_rect = [ax1_left, bottom, ax1_width, height]
            ax1 = self.figure.add_axes(ax1_rect)
            ax1.id = i * 2
            self._axes.append(ax1)

            ax2_rect = [ax2_left, bottom, ax2_width, height]
            ax2 = self.figure.add_axes(ax2_rect, frameon=False)
            ax2.id = i * 2 + 1
            ax2.set_axis_off()
            self._axes.append(ax2)

            self.bfly_axes.append(ax1)
            self.topo_axes.append(ax2)
            self._topoax_data.append((ax2, layers))

            # plot data
            p = _utsnd._ax_butterfly(ax1, layers, mark, False, False, color,
                                     vlims)
            self.bfly_plots.append(p)

            self._xvalues = np.union1d(self._xvalues, p._xvalues)

            # find and print epoch title
            for l in layers:
                if l.name is None:
                    continue
                y_text = bottom + y_sep / 2
                ax1.text(x_text, y_text, l.name,
                         transform=self.figure.transFigure,
                         ha='center', va='center', rotation='vertical')
                break

        self._configure_xaxis_dim('time', xlabel, xticklabels, self.bfly_axes)
        self._configure_yaxis(epochs[0][0], ylabel, self.bfly_axes)
        for ax in self.bfly_axes[:-1]:
            ax.xaxis.set_ticklabels(())

        # setup callback
        self.canvas.mpl_connect('button_press_event', self._on_click)
        self.canvas.mpl_connect('key_press_event', self._on_key)
        self._realtime_topo = True
        self._t_label = None
        self._frame.store_canvas()
        self._draw_topo(0, draw=False)
        self._show()

    def _draw_topo(self, t, draw=True):
        self._current_t = t
        epochs = [[l.sub(time=t) for l in layers]
                  for layers in self._epochs]

        if not self.topo_plots:
            for ax, layers in zip(self.topo_axes, epochs):
                p = _ax_topomap(ax, layers, vlims=self._vlims,
                                **self._topo_kwargs)
                self.topo_plots.append(p)
#             self._t_label = ax.text(.5, -0.1, t_str, ha='center', va='top')
        else:
            for layers, p in zip(epochs, self.topo_plots):
                p.set_data(layers)
#             self._t_label.set_text(t_str)

        if draw:
            self._frame.redraw(axes=self.topo_axes)  # , artists=[self._t_label])  # , artists=self.t_markers)

    def _rm_t_markers(self):
        "Remove markers of a specific time point (vlines and t-label)"
        if self._t_label:
            self._t_label.remove()
            self._t_label = None

        if self.t_markers:
            for m in self.t_markers:
                m.remove()
            self.t_markers = []

    def set_topo_t(self, t):
        "set the time point of the topo-maps"
        self._realtime_topo = False
        self._draw_topo(t, draw=False)

        # update t-markers
        self._rm_t_markers()
        for ax in self.bfly_axes:
            t_marker = ax.axvline(t, color='k')
            self.t_markers.append(t_marker)

        # add time label
        ax = self.topo_axes[-1]
        self._t_label = ax.text(.5, -0.1, "t = %s ms" % round(t * 1e3),
                                ha='center', va='top')

        self.canvas.draw()  # otherwise time label does not get redrawn

    def _on_click(self, event):
        ax = event.inaxes
        if ax in self.bfly_axes:
            button = {1:'l', 2:'r', 3:'r'}[event.button]
            if button == 'l':
                t = event.xdata
                self.set_topo_t(t)
            elif (button == 'r') and (self._realtime_topo == False):
                self._rm_t_markers()
                self._realtime_topo = True
                self.canvas.draw()
#                self._frame.redraw(axes=self.bfly_axes) # this leaves the time label

    def _on_key(self, event):
        ax = event.inaxes
        key = event.key
        if key == 't':
            if ax in self.bfly_axes:
                p = self.bfly_plots[ax.id // 2]
                t = event.xdata
                seg = [l.sub(time=t) for l in p.data]
                Topomap(seg)
            elif ax in self.topo_axes:
                p = self.topo_plots[ax.id // 2]
                Topomap(p.data)
        elif key == 'right' and not self._realtime_topo:
            i = np.where(self._xvalues > self._current_t)[0][0]
            t = self._xvalues[i]
            self.set_topo_t(t)
        elif key == 'left' and not self._realtime_topo:
            i = np.where(self._xvalues < self._current_t)[0][-1]
            t = self._xvalues[i]
            self.set_topo_t(t)

    def _on_leave_axes(self, event):
        "update the status bar when the cursor leaves axes"
        txt = "Topomap: t = %.3f" % self._current_t
        self._frame.SetStatusText(txt)

    def _on_motion(self, event):
        "update the status bar for mouse movement"
        super(self.__class__, self)._on_motion(event)
        ax = event.inaxes
        if ax in self.bfly_axes and self._realtime_topo:
            self._draw_topo(event.xdata)

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
        for p in self.topo_plots:
            p.add_contour(meas, level, color)
        self.draw()

    def get_vlim(self):
        "Retrieve ``(vmax, vmin)`` tuple of the first plot"
        p = self.bfly_plots[0]
        return p.vmax, p.vmin

    def set_cmap(self, cmap):
        "Change the colormap"
        for p in self.topo_plots:
            p.set_cmap(cmap)
        self.draw()

    def set_vlim(self, vmax=None, vmin=None):
        """Change the range of values displayed in butterfly-plots.
        """
        for topo, bfly in izip(self.topo_plots, self.bfly_plots):
            topo.set_vlim(vmax, vmin=vmin)
            kwa = topo.layers[0].get_kwargs()
            bfly.set_vlim(kwa['vmax'], vmin=kwa['vmin'])

        self.canvas.draw()


class _plt_topomap(_utsnd._plt_im_array):
    def __init__(self, ax, ndvar, overlay, proj='default', res=100,
                 interpolation=None, im_frame=0.02, vlims={}, cmaps={},
                 contours={}, method='linear'):
        """
        Parameters
        ----------
        im_frame : scalar
            Empty space beyond outmost sensors in the im plot.
        vmax : scalar
            Override the colorspace vmax.
        method : 'nearest' | 'linear' | 'cubic' | 'spline'
            Method for interpolating topo-map between sensors.
        """
        im_kwa = _base.find_im_args(ndvar, overlay, vlims, cmaps)
        self._contours = _base.find_ct_args(ndvar, overlay, contours)
        self._meas = ndvar.info.get('meas', _base.default_meas)

        emin = -im_frame
        emax = 1 + im_frame
        extent = (emin, emax, emin, emax)

        # store attributes
        self.ax = ax
        self.cont = None
        self._aspect = 'equal'
        self._extent = extent
        self._proj = proj
        self._grid = np.linspace(emin, emax, res)
        self._mgrid = tuple(np.meshgrid(self._grid, self._grid))
        self._method = method

        data = self._data_from_ndvar(ndvar)
        if im_kwa is not None:
            self.im = ax.imshow(data, extent=extent, origin='lower',
                                interpolation=interpolation, **im_kwa)
            self._cmap = im_kwa['cmap']
        else:
            self.im = None

        # draw flexible part
        self._data = data
        self._draw_contours()

    def _data_from_ndvar(self, ndvar):
        v = ndvar.get_data(('sensor',))
        locs = ndvar.sensor.get_locs_2d(self._proj)
        if self._method == 'spline':
            k = int(floor(sqrt(len(locs)))) - 1
            tck = interpolate.bisplrep(locs[:, 1], locs[:, 0], v, kx=k, ky=k)
            return interpolate.bisplev(self._grid, self._grid, tck)
        else:
            isnan = np.isnan(v)
            if np.any(isnan):
                nanmap = interpolate.griddata(locs, isnan, self._mgrid, self._method)
                mask = nanmap > 0.5
                v = np.where(isnan, 0, v)
                vmap = interpolate.griddata(locs, v, self._mgrid, self._method)
                np.place(vmap, mask, np.NaN)
                return vmap
            return interpolate.griddata(locs, v, self._mgrid, self._method)


class _ax_topomap(_utsnd._ax_im_array):

    def __init__(self, ax, layers, title=True, sensorlabels=None, mark=None,
                 mcolor=None, proj='default', res=100, interpolation=None,
                 xlabel=None, im_frame=0.02, vlims={}, cmaps={}, contours={}):
        """
        Parameters
        ----------
        sensorlabels : None | 'index' | 'name' | 'fullname'
            Show sensor labels. For 'name', any prefix common to all names
            is removed; with 'fullname', the full name is shown.
        mark : list of IDs
            highlight a subset of the sensors

        """
        self.ax = ax
        self.data = layers
        self.layers = []

        ax.set_axis_off()
        overlay = False
        for layer in layers:
            h = _plt_topomap(ax, layer, overlay, proj, res, interpolation,
                             im_frame, vlims, cmaps, contours)
            self.layers.append(h)
            if title is True:
                title = layer.name

            if xlabel is True:
                xlabel = layer.name

            overlay = True

        # plot sensors
        sensor_dim = layers[0].sensor
        self.sensors = _plt_map2d(ax, sensor_dim, proj, labels=sensorlabels)

        if mark is not None:
            sensor_dim = layers[0].sensor
            kw = {'marker': '.', 'ms': 3, 'markeredgewidth': 1,'ls': ''}
            if mcolor is not None:
                kw['color'] = mcolor

            _plt_map2d(ax, sensor_dim, proj, mark=mark, kwargs=kw)


        ax.set_xlim(-im_frame, 1 + im_frame)
        ax.set_ylim(-im_frame, 1 + im_frame)

        if isinstance(xlabel, basestring):
            x, y = ax.transData.inverted().transform(ax.transAxes.transform((0.5, 0)))
            ax.text(x, y, xlabel, ha='center', va='top')

        if isinstance(title, basestring):
            self.title = ax.set_title(title)
        else:
            self.title = None


class _TopoWindow:
    """Helper class for TopoArray.

    Maintains a topomap corresponding to one segment with flexible time point.
    """
    def __init__(self, ax, parent, **plot_args):
        self.ax = ax
        self.parent = parent
        self.plot_args = plot_args
        # initial plot state
        self.t_line = None
        self.pointer = None
        self.plot = None

    def update(self, t=None):
        if t is not None:
            if self.t_line:
                self.t_line.remove()
            self.t_line = self.parent.ax.axvline(t, c='r')

            t_str = "%i ms" % round(t * 1e3)
            if self.pointer:
                self.pointer.set_axes(self.parent.ax)
                self.pointer.xy = (t, 1)
                self.pointer.set_text(t_str)
                self.pointer.set_visible(True)
            else:
                xytext = self.ax.transAxes.transform((.5, 1))
                # These coordinates are in 'figure pixels'. They do not scale
                # when the figure is rescaled, so we need to transform them
                # into 'figure fraction' coordinates
                inv = self.ax.figure.transFigure.inverted()
                xytext = inv.transform(xytext)
                self.pointer = self.parent.ax.annotate(t_str, (t, 0),
                                    xycoords='data',
                                    xytext=xytext,
                                    textcoords='figure fraction',
                                    horizontalalignment='center',
                                    verticalalignment='center',
                                    arrowprops={'arrowstyle': '-',
                                                'shrinkB': 0,
                                                'connectionstyle': "angle3,angleA=90,angleB=0",
                                                'color': 'r'},
#                                    arrowprops={'width':1, 'frac':0,
#                                                'headwidth':0, 'color':'r',
#                                                'shrink':.05},
                                    zorder=99)

            layers = [l.sub(time=t) for l in self.parent.data]
            if self.plot is None:
                self.plot = _ax_topomap(self.ax, layers, title=False,
                                        **self.plot_args)
            else:
                self.plot.set_data(layers)

    def clear(self):
        self.ax.cla()
        self.ax.set_axis_off()
        self.plot = None
        if self.t_line:
            self.t_line.remove()
            self.t_line = None
        if self.pointer:
            self.pointer.remove()
            self.pointer = None


class TopoArray(_EelFigure):
    """Channel by sample plots with topomaps for individual time points

    Parameters
    ----------
    epochs :
        Epoch(s) to plot.
    Xax : None | categorial
        Create a separate plot for each cell in this model.
    title : None | string
        Figure title.
    ntopo | int
        number of topomaps per array-plot.
    t : list of scalar (len <= ntopo)
        Time points for topomaps.
    ds : None | Dataset
        If a Dataset is provided, ``epochs`` and ``Xax`` can be specified
        as strings.
    vmax, vmin : None | scalar
        Override the default plot limits. If only vmax is specified, vmin
        is set to -vmax.
    xticklabels : bool
        Add tick-labels to the x-axis (default True).

    Notes
    -----
     - LMB click on a topomap selects it for tracking the mouse pointer
         - LMB on the array plot fixates the topomap time point
     - RMB on a topomap removes the topomap

    """
    _make_axes = False

    def __init__(self, epochs, Xax=None, title=None, ntopo=3, t=[], ds=None,
                 vmax=None, vmin=None, xticklabels=True, *args, **kwargs):
        epochs, _ = _base.unpack_epochs_arg(epochs, ('time', 'sensor'), Xax, ds)
        n_epochs = len(epochs)
        n_topo_total = ntopo * n_epochs

        # create figure
        _EelFigure.__init__(self, 'TopoArray Plot', n_epochs, 6, 1.5, False, None,
                            *args, **kwargs)
        fig = self.figure
        axw = self._layout.axw

        # fig coordinates
        x_frame_l = .6 / axw / n_epochs
        x_frame_r = .025 / n_epochs
        x_sep = .01 / n_epochs

        x_per_ax = (1 - x_frame_l - x_frame_r) / n_epochs


        fig.subplots_adjust(left=x_frame_l,
                            bottom=.05,
                            right=1 - x_frame_r,
                            top=.9,
                            wspace=.1, hspace=.3)
        if title:
            fig.suptitle(title)
        self.title = title

        vlims = _base.find_fig_vlims(epochs, True, vmax, vmin)

        # save important properties
        self._epochs = epochs
        self._ntopo = ntopo
        self._vlims = vlims  # keep track of these for replotting topomaps
        self._cmaps = {}
        self._contours = {}
        self._default_xlabel_ax = -1 - ntopo

        # im_array plots
        self._array_axes = []
        self._array_plots = []
        self._topo_windows = []
        ax_height = .4 + .07 * (not title)
        ax_bottom = .45  # + .05*(not title)
        for i, layers in enumerate(epochs):
            ax_left = x_frame_l + i * (x_per_ax + x_sep)
            ax_right = 1 - x_frame_r - (n_epochs - i - 1) * (x_per_ax + x_sep)
            ax_width = ax_right - ax_left
            ax = fig.add_axes((ax_left, ax_bottom, ax_width, ax_height),
                              picker=True)
            ax.ID = i
            ax.type = 'main'
            im_plot = _utsnd._ax_im_array(ax, layers, vlims=vlims)
            self._axes.append(ax)
            self._array_axes.append(ax)
            self._array_plots.append(im_plot)
            if i > 0:
                ax.yaxis.set_visible(False)

            # topo plots
            for j in range(ntopo):
                ID = i * ntopo + j
                ax = fig.add_subplot(3, n_topo_total, 2 * n_topo_total + 1 + ID,
                                     picker=True, xticks=[], yticks=[])
                ax.ID = ID
                ax.type = 'window'
                win = _TopoWindow(ax, im_plot, vlims=self._vlims,
                                  cmaps=self._cmaps, contours=self._contours)
                self._axes.append(ax)
                self._topo_windows.append(win)

        # if t argument is provided, set topo-map time points
        if t:
            if np.isscalar(t):
                t = [t]
            self.set_topo_ts(*t)

        self._configure_xaxis_dim('time', True, xticklabels, self._array_axes)
        self._configure_yaxis_dim('sensor', True, self._array_axes)

        # setup callback
        self._selected_window = None
        self.canvas.mpl_connect('pick_event', self._pick_handler)
        self.canvas.mpl_connect('motion_notify_event', self._motion_handler)
        self._frame.store_canvas()
        self._show()

    def __repr__(self):
        e_repr = []
        for e in self._epochs:
            if hasattr(e, 'name'):
                e_repr.append(e.name)
            else:
                e_repr.append([ie.name for ie in e])
        kwargs = {'s': repr(e_repr),
                  't': ' %r' % self.title if self.title else ''}
        txt = "<plot.TopoArray{t} ({s})>".format(**kwargs)
        return txt

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
        self._contours.setdefault(meas, {})[level] = color
        for p in self._iter_plots():
            p.add_contour(meas, level, color)
        self.draw()

    def _iter_plots(self):
        "Iterate through non-empty plots"
        for p in self._array_plots:
            yield p
        for w in self._topo_windows:
            if w.plot is not None:
                yield w.plot

    def set_cmap(self, cmap, meas=None):
        """Change the colormap

        Parameters
        ----------
        cmap : str | colormap
            New colormap.
        meas : None | str
            Measurement to which to apply the colormap. With None, it is
            applied to all.
        """
        self._cmaps[meas] = cmap
        for p in self._iter_plots():
            p.set_cmap(cmap, meas)
        self.draw()

    def set_topo_t_single(self, topo_id, t, parent_im_id='auto'):
        """
        Set the time for a single topomap.

        Parameters
        ----------
        topo_id : int
            Index of the topomap (numbered throughout the figure).
        t : scalar or ``None``
            time point; ``None`` clears the topomap
        parent_im_id : 'auto' | int
            Index of the array plot from which to draw the topo plot. For
            'auto', the array plot above the topomap is used.
        """
        # get parent ax
        if parent_im_id == 'auto':
            parent_im_id = int(topo_id / self._ntopo)
        # get window ax
        w = self._topo_windows[topo_id]
        w.clear()

        if t is not None:
            w.update(t=t)

        self.canvas.draw()

    def set_topo_t(self, topo_id, t):
        """
        Set the time point for a topo-map (same for all array plots).

        Parameters
        ----------
        topo_id : int
            Index of the topomap (numberd for each array-plot).
        t : scalar or ``None``
            time point; ``None`` clears the topomap

        See Also
        --------
        .set_topo_ts : set several topomap time points at once
        .set_topo_t_single : set the time point of a single topomap
        """
        for i in xrange(len(self._array_plots)):
            _topo = self._ntopo * i + topo_id
            self.set_topo_t_single(_topo, t, parent_im_id=i)

    def set_topo_ts(self, *t_list):
        """
        Set the time points for several topomaps (with ts identical for the
        different array plots).
        """
        for i, t in enumerate(t_list):
            self.set_topo_t(i, t)

    def _window_update(self, mouseevent):
        "update a window (used for mouse-over and for pick)"
        t = mouseevent.xdata
        self._selected_window.update(t=t)
        self._frame.redraw(axes=[self._selected_window.ax])

    def _pick_handler(self, pickevent):
        mouseevent = pickevent.mouseevent
        ax = pickevent.artist
        if ax.type == 'window':
            button = mouseevent.button  # 1: Left
            window = self._topo_windows[ax.ID]
            if button == 1:
                self._selected_window = window
            elif button in (2, 3):
                Id = window.ax.ID % self._ntopo
                self.set_topo_t(Id, None)
            else:
                pass
        elif (ax.type == 'main') and (self._selected_window != None):
            self._selected_window.clear()  # to side track pdf export transparency issue
#            self._window_update(mouseevent, ax)

            # update corresponding topo_windows
            t = mouseevent.xdata
            Id = self._selected_window.ax.ID % self._ntopo
            self.set_topo_t(Id, t)

            self._selected_window = None
            self.canvas.draw()

    def _motion_handler(self, mouseevent):
        ax = mouseevent.inaxes
        if getattr(ax, 'type', None) == 'main':
            if self._selected_window != None:
                self._window_update(mouseevent)
