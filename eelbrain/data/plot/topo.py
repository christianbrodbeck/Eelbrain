"""
plot.topo
=========

Plots with topographic maps.

"""
from __future__ import division
from itertools import chain

import numpy as np

from . import _base
from . import utsnd as _utsnd
from .sensors import _plt_map2d

# try:
#    from _topo3d import *
# except:
#    logging.info("eelbrain.plot.topo: _topo3d import failed, 3d-plots not available")


__hide__ = ['cs', 'test', 'utsnd']



class topomap(_base.eelfigure):
    "Plot individual topogeraphies"
    def __init__(self, epochs, Xax=None, sensors=True, proj='default',
                 vmax=None, title=None, res=100, interpolation='nearest',
                 ds=None, **layout):
        """
        Plot individual topogeraphies

        Parameters
        ----------
        epochs : ndvar | list of ndvar, dims = ([case,] sensor,)
            Data to plot.
        sensors : None | 'idx' | 'name' | 'fullname'
            Show sensor labels. For 'name', any prefix common to all names
            is removed; with 'fullname', the full name is shown.
        proj : str
            The sensor projection to use for topomaps.
        vmax : scalar
            Vmax for plots.
        size : scalar
            Side length in inches of individual axes.
        dpi : scalar
            Dpi of the figure.
        title : str
            Title (shown in the window, not figure title).
        res : int
            Resolution of the topomaps (width = height = ``res``).
        interpolation : str
            Matplotlib imshow() parameter for topomaps.
        """
        epochs = self.epochs = _base.unpack_epochs_arg(epochs, 1, Xax, ds)
        nax = len(epochs)
        super(topomap, self).__init__("plot.topo.topomap", nax, layout, 1, 7,
                                      figtitle=title)

        topo_kwargs = {'res': res,
                       'interpolation': interpolation,
                       'proj': proj,
                       'sensors': sensors,
                       'vlims': _base.find_fig_vlims(epochs)}

        self._topomaps = []
        for i, ax, layers in self._iter_ax(epochs):
            ax.ID = i
            h = _ax_topomap(ax, layers, title=True, **topo_kwargs)
            self._topomaps.append(h)

        self._label_color = 'k'
        if isinstance(sensors, str):
            self.set_label_text(sensors)

        self._show()

    def _fill_toolbar(self, tb):
        import wx
        tb.AddSeparator()

        # sensor labels
        lbl = wx.StaticText(tb, -1, "Labels:")
        tb.AddControl(lbl)
        choice = wx.Choice(tb, -1, choices=['None', 'Index', 'Name'])
        tb.AddControl(choice)
        self._SensorLabelChoice = choice
        choice.Bind(wx.EVT_CHOICE, self._OnSensorLabelChoice)

        # sensor label color
        choices = ['black', 'white', 'blue', 'green', 'red', 'cyan', 'magenta',
                   'yellow']
        choice = wx.Choice(tb, -1, choices=choices)
        tb.AddControl(choice)
        self._SensorLabelColorChoice = choice
        choice.Bind(wx.EVT_CHOICE, self._OnSensorLabelColorChoice)

        btn = wx.Button(tb, label="Mark")  # , style=wx.BU_EXACTFIT)
        btn.Bind(wx.EVT_BUTTON, self._OnMarkSensor)
        tb.AddControl(btn)

    def _OnMarkSensor(self, event):
        import wx
        msg = "Channels to mark, separated by comma"
        dlg = wx.TextEntryDialog(self._frame, msg, "Mark Sensor")
        if dlg.ShowModal() != wx.ID_OK:
            return

        chs = filter(None, map(unicode.strip, dlg.GetValue().split(',')))
        try:
            self.mark_sensors(chs)
        except Exception as exc:
            msg = '%s: %s' % (type(exc).__name__, exc)
            sty = wx.OK | wx.ICON_ERROR
            wx.MessageBox(msg, "Mark Sensors Failed for %r" % chs, style=sty)

    def _OnSensorLabelColorChoice(self, event):
        sel = event.GetSelection()
        color = ['k', 'w', 'b', 'g', 'r', 'c', 'm', 'y'][sel]
        self.set_label_color(color)

    def _OnSensorLabelChoice(self, event):
        sel = event.GetSelection()
        text = [None, 'idx', 'name'][sel]
        self.set_label_text(text)

    def mark_sensors(self, sensors, marker='bo'):
        for p in self._topomaps:
            p.sensors.mark_sensors(sensors, marker)
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
        for p in self._topomaps:
            p.set_cmap(cmap, base, overlays)
        self.draw()

    def set_label_color(self, color='w'):
        if hasattr(self, '_SensorLabelChoice'):
            sels = ['k', 'w', 'b', 'g', 'r', 'c', 'm', 'y']
            if color in sels:
                sel = sels.index(color)
                self._SensorLabelColorChoice.SetSelection(sel)

        self._label_color = color
        for p in self._topomaps:
            p.sensors.set_label_color(color)
        self.draw()

    def set_label_text(self, text='idx'):
        """Add/remove sensor labels

        Parameters
        ----------
        labels : None | 'idx' | 'name' | 'fullname'
            Content of the labels. For 'name', any prefix common to all names
            is removed; with 'fullname', the full name is shown.
        """
        if hasattr(self, '_SensorLabelChoice'):
            sel = [None, 'idx', 'name'].index(text)
            self._SensorLabelChoice.SetSelection(sel)

        for p in self._topomaps:
            p.sensors.show_labels(text, color=self._label_color)
        self.draw()

    def set_vlim(self, vmax=None, meas=None, vmin=None):
        "Change the range of the data shown in he colormap"
        for p in self._topomaps:
            p.set_vlim(vmax, meas, vmin)
        self.draw()


class butterfly(_base.eelfigure):
    """
    Butterfly plot with corresponding topomaps.

     - LMB click in butterfly plots fixates the topomap time.
     - RMB click in butterfly plots removes the time point, the topomaps follow
       the mouse pointer.

    """
    def __init__(self, epochs, Xax=None, title=None, xlabel=True, ylabel=True,
                 proj='default', res=100, interpolation='nearest', color=None,
                 sensors=True, ROI=None, ds=None, axh=3, ax_aspect=2,
                 **fig_kwa):
        """
        Parameters
        ----------

        ROI : list of indices
            plot a subset of sensors
        sensors : bool
            determines whether all sensors are marked in the topo-maps

        **Figure Layout:**

        size : scalar
            in inches: height of the butterfly axes as well as side length of
            the topomap axes
        ax_aspect : scalar
            multiplier for the width of butterfly plots based on their height
        res : int
            resolution of the topomap plots (res x res pixels)
        interpolation : 'nearest' | ...
            matplotlib imshow kwargs
        vmax : None | scalar
            Override the default plot limits.

        """
        epochs = self.epochs = _base.unpack_epochs_arg(epochs, 2, Xax, ds)
        n_plots = len(epochs)

        # create figure
        x_size = axh * (1 + ax_aspect)
        y_size = axh * n_plots

        fig_kwa.update(figsize=(x_size, y_size))
        super(butterfly, self).__init__("plot.topo.butterfly", None,
                                        fig_kwa=fig_kwa, figtitle=title)

        # axes sizes
        frame = .05  # in inches; .4

        xframe = frame / x_size
        x_left_ylabel = 0.5 / x_size if ylabel else 0
        x_left_title = 0.5 / x_size
        x_text = x_left_title / 3
        ax1_left = xframe + x_left_title + x_left_ylabel
        ax1_width = ax_aspect / (ax_aspect + 1) - ax1_left - xframe / 2
        ax2_left = ax_aspect / (ax_aspect + 1) + xframe / 2
        ax2_width = 1 / (ax_aspect + 1) - 1.5 * xframe

        yframe = frame / y_size
        y_bottomframe = 0.5 / y_size
#        y_bottom = yframe
        y_sep = (1 - y_bottomframe) / n_plots
        height = y_sep - yframe

        self.topo_kwargs = {'proj': proj,
                            'res': res,
                            'interpolation': interpolation,
                            'sensors': sensors,
                            'ROI': ROI,
                            'ROIcolor': color,
                            'title': False}

        t = 0
        self.bfly_axes = []
        self.topo_axes = []
        self.bfly_plots = []
        self.topo_plots = []
        self._topoax_data = []
        self.t_markers = []
        vlims = _base.find_fig_vlims(epochs, True)

        # plot epochs (x/y are in figure coordinates)
        for i, layers in enumerate(epochs):
            # position axes
            bottom = 1 - y_sep * (1 + i)

            ax1_rect = [ax1_left, bottom, ax1_width, height]
            ax2_rect = [ax2_left, bottom, ax2_width, height]
            ax1 = self.figure.add_axes(ax1_rect)
            ax1.ID = i

            ax2 = self.figure.add_axes(ax2_rect, frameon=False)
            ax2.set_axis_off()

            # t - label
            if len(self.topo_axes) == n_plots - 1:
                self._t_title = ax2.text(.0, 0, 't = %.3f' % t, ha='center')

            self.bfly_axes.append(ax1)
            self.topo_axes.append(ax2)
            self._topoax_data.append((ax2, layers))

            show_x_axis = (i == n_plots - 1)

            # plot data
            p = _utsnd._ax_butterfly(ax1, layers, sensors=ROI, title=False,
                                    xlabel=show_x_axis, ylabel=ylabel,
                                    color=color, vlims=vlims)
            self.bfly_plots.append(p)

            if not show_x_axis:
                ax1.xaxis.set_ticklabels([])

            # find and print epoch title
            title = True
            for l in layers:
                if title is True:
                    title = getattr(l, 'name', True)
            if isinstance(title, str):
                y_text = bottom + y_sep / 2
                ax1.text(x_text, y_text, title, transform=self.figure.transFigure,
                         ha='center', va='center', rotation='vertical')

        # setup callback
        self.canvas.mpl_connect('button_press_event', self._on_click)
        self._realtime_topo = True
        self._frame.store_canvas()
        self._draw_topo(0, draw=False)
        self._show(tight=False)

    def _draw_topo(self, t, draw=True):
        self._current_t = t
        del self.topo_plots[:]
        for ax, layers, p in zip(self.topo_axes, self.epochs, self.bfly_plots):
            ax.cla()
            layers = [l.subdata(time=t) for l in layers]
            p = _ax_topomap(ax, layers, vmin=p.vmin, vmax=p.vmax,
                            **self.topo_kwargs)
            self.topo_plots.append(p)

        if draw:
            self._frame.redraw(axes=self.topo_axes)  # , artists=self.t_markers)

    def _rm_markers(self):
        if self.t_markers:
            for m in self.t_markers:
                m.remove()
            self.t_markers = []

    def set_topo_t(self, t):
        "set the time point of the topo-maps"
        self._realtime_topo = False
        self._draw_topo(t, draw=False)

        # update t-markers
        self._rm_markers()
        for ax in self.bfly_axes:
            t_marker = ax.axvline(t, color='k')
            self.t_markers.append(t_marker)

        # add time label
        ax = self.topo_axes[-1]
        t_str = "t = %.3f" % t
        self._t_label = ax.text(.5, -0.1, t_str, ha='center', va='top')

        self.canvas.draw()  # otherwise time label does not get redrawn

    def _on_click(self, event):
        ax = event.inaxes
        if ax and hasattr(ax, 'ID'):
            button = {1:'l', 2:'r', 3:'r'}[event.button]
            if button == 'l':
                t = event.xdata
                self.set_topo_t(t)
            elif (button == 'r') and (self._realtime_topo == False):
                self._rm_markers()
                self._t_label.remove()
                self._realtime_topo = True
                self.canvas.draw()
#                self._frame.redraw(axes=self.bfly_axes) # this leaves the time label

    def _on_leave_axes(self, event):
        "update the status bar when the cursor leaves axes"
        txt = "Topomap: t = %.3f" % self._current_t
        self._frame.SetStatusText(txt)

    def _on_motion(self, event):
        "update the status bar for mouse movement"
        ax = event.inaxes
        if ax and hasattr(ax, 'ID'):
            super(self.__class__, self)._on_motion(event)
        if self._realtime_topo and ax and hasattr(ax, 'ID'):
            self._draw_topo(event.xdata)

    def set_cmap(self, cmap, base=True, overlays=False, **kwa):
        "Change the colormap"
        for p in self.topo_plots:
            p.set_cmap(cmap, base, overlays)
        self.topo_kwargs['cmap'] = cmap
        self.draw()


    def set_vlim(self, vmax=None, vmin=None):
        """Change the range of values displayed in butterfly-plots.
        """
        for topo, bfly in zip(self.topo_plots, self.bfly_plots):
            topo.set_vlim(vmax, vmin=vmin)
            kwa = topo.layers[0].get_kwargs()
            bfly.set_vlim(kwa['vmax'], vmin=kwa['vmin'])

        self.canvas.draw()


class _plt_topomap(_utsnd._plt_im_array):
    def __init__(self, ax, ndvar, overlay, proj='default', res=100,
                 im_frame=0.02, colorspace=None, vlims={}, **im_kwargs):
        """
        Parameters
        ----------
        im_frame : scalar
            Empty space beyond outmost sensors in the im plot.
        vmax : scalar
            Override the colorspace vmax.
        """
        im_kwa = _base.find_im_args(ndvar, overlay, vlims)
        ct_kwa = _base.find_ct_args(ndvar, overlay)
        self._meas = ndvar.info.get('meas', _base.default_meas)
        self.ax = ax

        Y = ndvar.get_data(('sensor',))
        Ymap = ndvar.sensor.get_im_for_topo(Y, proj=proj, res=res, frame=im_frame)

        emin = -im_frame
        emax = 1 + im_frame
        map_kwargs = {'origin': 'lower', 'extent': (emin, emax, emin, emax)}

        if im_kwa is not None:
            im_kwa.update(map_kwargs)
            im_kwa.update(im_kwargs)
            self.im = ax.imshow(Ymap, **im_kwa)
            self._cmap = im_kwa['cmap']
        else:
            self.im = None

        # contours
        if ct_kwa is not None:
            ct_kwa.update(map_kwargs)
            self.contour = ax.contour(Ymap, **ct_kwa)
        else:
            self.contour = None


class _ax_topomap:
    def __init__(self, ax, layers, title=True, sensors=None, ROI=None,
                 ROIcolor=True, proj='default', xlabel=None, im_frame=0.02,
                 vlims={}, **im_kwargs):
        """
        Parameters
        ----------
        sensors : bool | str
            plot sensor markers (str to add label:
        ROI : list of IDs
            highlight a subset of the sensors

        """
        self.ax = ax
        self.layers = []

        ax.set_axis_off()
        overlay = False
        for layer in layers:
            h = _plt_topomap(ax, layer, overlay, im_frame=im_frame, proj=proj,
                             vlims=vlims, **im_kwargs)
            self.layers.append(h)
            if title is True:
                title = getattr(layer, 'name', True)
            overlay = True

        # plot sensors
        if sensors:
            sensor_dim = layers[0].sensor
            self.sensors = _plt_map2d(ax, sensor_dim, proj=proj)
            if isinstance(sensors, str):
                text = sensors
            else:
                text = None
            self.sensors.show_labels(text=text)

        if ROI is not None:
            sensor_dim = layers[0].sensor
            kw = dict(marker='.',  # symbol
                    ms=3,  # marker size
                    markeredgewidth=1,
                    ls='')

            if ROIcolor is not True:
                kw['color'] = ROIcolor

            _plt_map2d(ax, sensor_dim, proj=proj, ROI=ROI, kwargs=kw)


        ax.set_xlim(-im_frame, 1 + im_frame)
        ax.set_ylim(-im_frame, 1 + im_frame)

        if isinstance(xlabel, str):
            ax.set_xlabel(xlabel)

        if isinstance(title, str):
            self.title = ax.set_title(title)
        else:
            self.title = None

    def set_cmap(self, cmap, base=True, overlays=False, **kwa):
        """Change the colormap in the topomap

        Parameters
        ----------
        cmap : str | colormap
            New colormap.
        base : bool
            Apply the new colormap in the lowest layer of each plot.
        overlays : bool
            Apply the new colormap to the layers above the first layer.
        """
        if base and overlays:
            layers = self.layers
        elif base:
            layers = self.layers[:1]
        elif overlays:
            layers = self.layers[1:]
        else:
            return

        for l in layers:
            l.set_cmap(cmap)

    def set_vlim(self, vmax=None, meas=None, vmin=None):
        for l in self.layers:
            l.set_vlim(vmax, meas, vmin)


class _Window_Topo:
    """Helper class for array"""
    def __init__(self, ax, parent):
        self.ax = ax
        # initial plot state
        self.t_line = None
        self.pointer = None
        self.parent = parent
        self.plot = None

    def update(self, t=None):
        if t != None:
            if self.t_line:
                self.t_line.remove()
            self.t_line = self.parent.ax.axvline(t, c='r')
            # self.pointer.xy=(t,1)
            # self.pointer.set_text("t = %s"%t)
            if self.pointer:
                # print 't =', t
                self.pointer.set_axes(self.parent.ax)
                self.pointer.xy = (t, 1)
                self.pointer.set_text("t=%.3g" % t)
                self.pointer.set_visible(True)
            else:
                xytext = self.ax.transAxes.transform((.5, 1))
                # These coordinates are in 'figure pixels'. They do not scale
                # when the figure is rescaled, so we need to transform them
                # into 'figure fraction' coordinates
                inv = self.ax.figure.transFigure.inverted()
                xytext = inv.transform(xytext)
                self.pointer = self.parent.ax.annotate("t=%.3g" % t, (t, 0),
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

            self.ax.cla()
            layers = [l.subdata(time=t) for l in self.parent.data]
            self.plot = _ax_topomap(self.ax, layers, title=False,
                                    **self.parent.kwargs)

    def clear(self):
        self.ax.cla()
        self.ax.set_axis_off()
        if self.t_line:
            self.t_line.remove()
            self.t_line = None
        if self.pointer:
            self.pointer.remove()
            self.pointer = None




class array(_base.eelfigure):
    """
    Channel by sample plots with corresponding topomaps

     - LMB click on a topomap selects it for tracking the mouse pointer
         - LMB on the array plot fixates the topomap time point
     - RMB on a topomap removes the topomap

    """
    def __init__(self, epochs, Xax=None, title=None, axh=6, axw=5,
                 ntopo=3, t=[], ds=None,
                 **fig_kwa):
        """
        Channel by sample array-plots with topomaps corresponding to
        individual time points.

        Parameters
        ----------
        epochs :
            Epoch(s) to plot.
        title : str | None
            Figure title.
        height, width : scalar
            Axes height and width in inches.
        ntopo | int
            number of topomaps per array-plot.
        dpi : scalar
            Figure dpi.
        vmin, vmax : None | scalar
            Limit of the range of the data displayed.
        t : list of scalar (len <= ntopo)
            Time points for topomaps.
        """
        epochs = _base.unpack_epochs_arg(epochs, 2, Xax, ds)

        # figure properties
        n_epochs = len(epochs)
        n_topo_total = ntopo * n_epochs
        left_rim = axw / 4
        fig_width, fig_height = n_epochs * axw + left_rim, axh
        fig_kwa.update(figsize=(fig_width, fig_height))

        # fig coordinates
        x_frame_l = .6 / axw / n_epochs
        x_frame_r = .025 / n_epochs
        x_sep = .01 / n_epochs

        x_per_ax = (1 - x_frame_l - x_frame_r) / n_epochs

        # create figure
        super(array, self).__init__('plot.topo.array', None, fig_kwa=fig_kwa)
        fig = self.figure

        fig.subplots_adjust(left=x_frame_l,
                            bottom=.05,
                            right=1 - x_frame_r,
                            top=.9,
                            wspace=.1, hspace=.3)
        if title:
            fig.suptitle(title)
        self.title = title

        # im_array plots
        self.array_plots = []
        self.topo_windows = []
        vlims = _base.find_fig_vlims(epochs)
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
            self.array_plots.append(im_plot)
            if i > 0:
                ax.yaxis.set_visible(False)

            # topo plots
            for j in range(ntopo):
                ID = i * ntopo + j
                ax = fig.add_subplot(3, n_topo_total, 2 * n_topo_total + 1 + ID,
                                     picker=True, xticks=[], yticks=[])
                ax.ID = ID
                ax.type = 'window'
                self.topo_windows.append(_Window_Topo(ax, im_plot))

        # save important properties
        self.epochs = epochs
        self._ntopo = ntopo

        # if t argument is provided, set topo-pol time points
        if t:
            if np.isscalar(t):
                t = [t]
            self.setwins(*t)

        # setup callback
        self._selected_window = None
        self.canvas.mpl_connect('pick_event', self._pick_handler)
        self.canvas.mpl_connect('motion_notify_event', self._motion_handler)
        self._frame.store_canvas()
        self._show(tight=False)

    def __repr__(self):
        e_repr = []
        for e in self.epochs:
            if hasattr(e, 'name'):
                e_repr.append(e.name)
            else:
                e_repr.append([ie.name for ie in e])
        kwargs = {'s': repr(e_repr),
                  't': ' %r' % self.title if self.title else ''}
        txt = "<plot.topo.array{t} ({s})>".format(**kwargs)
        return txt

    def set_topo_single(self, topo, t, parent_im_id='auto'):
        """
        Set the time for a single topomap

        topo : int
            Id of the topomap (numbered throughout the figure).
        t : scalar or ``None``
            time point; ``None`` clears the topomap

        """
        # get parent ax
        if parent_im_id == 'auto':
            parent_im_id = int(topo / self._ntopo)
        # get window ax
        w = self.topo_windows[topo]
        w.clear()

        if t is not None:
            w.update(t=t)

        self.canvas.draw()

    def set_topowin(self, topo_id, t):
        """
        Set the time point for a topo-map (for all xbyx plots; In order to
        modify a single topoplot, use setone method).

        """
        for i in xrange(len(self.array_plots)):
            _topo = self._ntopo * i + topo_id
            self.set_topo_single(_topo, t, parent_im_id=i)

    def set_topowins(self, *t_list):
        """
        Set time points for several topomaps (calls self.set() for each value
        in t_list)

        """
        for i, t in enumerate(t_list):
            self.set_topowin(i, t)

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
            window = self.topo_windows[ax.ID]
            if button == 1:
                self._selected_window = window
            elif button in (2, 3):
                Id = window.ax.ID % self._ntopo
                self.set_topowin(Id, None)
            else:
                pass
        elif (ax.type == 'main') and (self._selected_window != None):
            self._selected_window.clear()  # to side track pdf export transparency issue
#            self._window_update(mouseevent, ax)

            # update corresponding topo_windows
            t = mouseevent.xdata
            Id = self._selected_window.ax.ID % self._ntopo
            self.set_topowin(Id, t)

            self._selected_window = None
            self.canvas.draw()

    def _motion_handler(self, mouseevent):
        ax = mouseevent.inaxes
        if getattr(ax, 'type', None) == 'main':
            if self._selected_window != None:
                self._window_update(mouseevent)


