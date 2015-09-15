'''
Plot sensor maps.
'''
# author: Christian Brodbeck


import os

import numpy as np
import matplotlib as mpl
from matplotlib.lines import Line2D

from .._data_obj import Datalist, as_sensor
from ._base import _EelFigure


# some useful kwarg dictionaries for different plot layouts
kwargs_mono = dict(mc='k',
                   lc='.5',
                   hllc='k',
                   hlmc='k',
                   hlms=7,
                   strlc='k')


class _plt_connectivity:
    def __init__(self, ax, locs, connectivity, linestyle={}):
        self.ax = ax
        self.locs = locs
        self._h = []
        self.show(connectivity, linestyle)

    def show(self, connectivity, linestyle={}):
        while self._h:
            self._h.pop().remove()

        if connectivity is None:
            return

        for c, r in connectivity:
            x = self.locs[[c, r], 0]
            y = self.locs[[c, r], 1]
            line = Line2D(x, y, **linestyle)
            self.ax.add_line(line)
            self._h.append(line)


class _ax_map2d:
    def __init__(self, ax, sensors, proj='default', extent=1,
                 frame=.02, kwargs=None):
        self.ax = ax

        if kwargs is None:
            kwargs = {'marker': 'x',  # symbol
                      'color': 'b',  # mpl plot kwargs ...
                      'ms': 3,  # marker size
                      'markeredgewidth': .5,
                      'ls': ''}

        ax.set_aspect('equal')
        # ax.set_frame_on(False)
        ax.set_axis_off()

        h = _plt_map2d(ax, sensors, proj, extent=extent, kwargs=kwargs)
        self.sensors = h

        locs = sensors.get_locs_2d(proj=proj, extent=extent)
        self.connectivity = _plt_connectivity(ax, locs, None)

        ax.set_xlim(-frame, 1 + frame)

    def remove(self):
        "remove from axes"
        self.sensors.remove()


class _plt_map2d:

    def __init__(self, ax, sensors, proj='default', extent=1, mark=None,
                 labels=None, kwargs=None):
        """
        Parameters
        ----------
        ax : matplotlib Axes
            Axes.
        sensors : Sensor
            Sensor dimension.

        labels : None | 'index' | 'name' | 'fullname'
            Content of the labels. For 'name', any prefix common to all names
            is removed; with 'fullname', the full name is shown.
        """
        self.ax = ax
        self.sensors = sensors
        self.locs = sensors.get_locs_2d(proj=proj, extent=extent)
        self._mark_handles = None

        if kwargs is None:
            kwargs = {'marker': '.',  # symbol
                      'color': 'k',  # mpl plot kwargs ...
                      'ms': 1,  # marker size
                      'markeredgewidth': .5,
                      'ls': ''}

        if mark is not None:
            mark_idx = sensors.dimindex(mark)
            locs = self.locs[mark_idx]
        else:
            locs = self.locs
            mark_idx = None

        self.mark = mark
        self._mark_idx = mark_idx
        self.markers = []
        self.labels = []
        if labels is not None:
            self.show_labels(labels)

        if 'color' in kwargs:
            h = ax.plot(locs[:, 0], locs[:, 1], **kwargs)
            self.markers += h
        else:
            colors = mpl.rcParams['axes.color_cycle']
            nc = len(colors)
            for i in xrange(len(locs)):
                kwargs['color'] = kwargs['mec'] = colors[i % nc]
                h = ax.plot(locs[i, 0], locs[i, 1], **kwargs)
                self.markers += h

    def mark_sensors(self, sensors, *args, **kwargs):
        """Mark specific sensors

        Parameters
        ----------
        sensors : None | Sensor dimension index
            Sensors which should be marked (None to clear all markings).
        others :
            Matplotlib :func:`pyplot.scatter` parameters for the marking
            sensors.
        """
        if sensors is None:
            while self._mark_handles:
                self._mark_handles.pop().remove()
            return
        elif not np.count_nonzero(sensors):
            return

        idx = self.sensors.dimindex(sensors)
        locs = self.locs[idx]
        self._mark_handles = self.ax.scatter(locs[:, 0], locs[:, 1], *args, **kwargs)

    def remove(self):
        "remove from axes"
        while self._mark_handles:
            self._mark_handles.pop().remove()
        while self.markers:
            self.markers.pop().remove()

    def show_labels(self, text='name', xpos=0, ypos=.01, **text_kwargs):
        """Plot labels for the sensors

        Parameters
        ----------
        text : None | 'index' | 'name' | 'fullname'
            Content of the labels. For 'name', any prefix common to all names
            is removed; with 'fullname', the full name is shown.
        xpos, ypos : scalar
            The position offset of the labels from the sensor markers.
        text_kwargs : **
            Matplotlib text parameters.
        """
        # remove existing labels
        while self.labels:
            self.labels.pop().remove()

        if not text:
            return

        kwargs = dict(color='k', fontsize=8, horizontalalignment='center',
                      verticalalignment='bottom')
        kwargs.update(text_kwargs)

        sensors = self.sensors
        if text == 'index':
            labels = map(str, xrange(len(sensors)))
        elif text == 'name':
            labels = sensors.names
            prefix = os.path.commonprefix(labels)
            pf_len = len(prefix)
            labels = [label[pf_len:] for label in labels]
        elif text == 'fullname':
            labels = sensors.names
        else:
            err = "text has to be 'index' or 'name', can't be %r" % text
            raise NotImplementedError(err)

        locs = self.locs
        if self._mark_idx is not None:
            labels = Datalist(labels)[self._mark_idx]
            locs = locs[self._mark_idx]

        locs = locs + [[xpos, ypos]]
        for loc, txt in zip(locs, labels):
            x , y = loc
            h = self.ax.text(x, y, txt, **kwargs)
            self.labels.append(h)

    def set_label_color(self, color='k'):
        """Change the color of all sensor labels

        Parameters
        ----------
        color : matplotlib color
            New color for the sensor labels.
        """
        for h in self.labels:
            h.set_color(color)


class SensorMapMixin:
    # expects self._sensor_plots to be list of _plt_map2d
    __label_options = ['None', 'Index', 'Name', 'Full Name']
    __label_option_args = [None, 'index', 'name', 'fullname']

    def __init__(self, sensor_plots, label=None):
        """Call after EelFigure init (toolbar fill)

        Parameters
        ----------
        sensor_plots : list of _plt_map2d
            Sensor-map objects.
        label : None | str
            Initial label argument (default None).
        """
        self.__label_color = 'k'
        self.__check_label_arg(label)
        self.__sensor_plots = sensor_plots
        self.__LabelChoice.SetSelection(self.__label_option_args.index(label))

    def _fill_toolbar(self, tb):
        import wx
        tb.AddSeparator()

        # sensor labels
        lbl = wx.StaticText(tb, -1, "Labels:")
        tb.AddControl(lbl)
        choice = wx.Choice(tb, -1, choices=self.__label_options)
        tb.AddControl(choice)
        self.__LabelChoice = choice
        choice.Bind(wx.EVT_CHOICE, self.__OnSensorLabelChoice)

        # sensor label color
        choices = ['black', 'white', 'blue', 'green', 'red', 'cyan', 'magenta',
                   'yellow']
        choice = wx.Choice(tb, -1, choices=choices)
        tb.AddControl(choice)
        self.__LabelColorChoice = choice
        choice.Bind(wx.EVT_CHOICE, self.__OnSensorLabelColorChoice)

        btn = wx.Button(tb, label="Mark")  # , style=wx.BU_EXACTFIT)
        btn.Bind(wx.EVT_BUTTON, self.__OnMarkSensor)
        tb.AddControl(btn)

    def __check_label_arg(self, arg):
        if arg not in self.__label_option_args:
            raise ValueError("Invalid sensor label argument: %s" % repr(arg))

    def __OnMarkSensor(self, event):
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

    def __OnSensorLabelChoice(self, event):
        sel = event.GetSelection()
        sel_arg = self.__label_option_args[sel]
        self.set_label_text(sel_arg)

    def __OnSensorLabelColorChoice(self, event):
        sel = event.GetSelection()
        color = ['k', 'w', 'b', 'g', 'r', 'c', 'm', 'y'][sel]
        self.set_label_color(color)

    def mark_sensors(self, sensors, *args, **kwargs):
        """Mark given sensors on the plots

        Parameters
        ----------
        sensors : None | Sensor dimension index
            Sensors which should be marked (None to clear all markings).
        s : scalar | sequence of scalars
            Marker size(s) in points^2.
        c : color | sequence of colors
            Marker color(s).
        marker : str
            Marker style, default: ``'o'``.

        (Matplotlib :func:`pyplot.scatter` parameters)
        """
        for p in self.__sensor_plots:
            p.mark_sensors(sensors, *args, **kwargs)
        self.draw()

    def set_label_color(self, color='w'):
        if hasattr(self, '_SensorLabelChoice'):
            sels = ['k', 'w', 'b', 'g', 'r', 'c', 'm', 'y']
            if color in sels:
                sel = sels.index(color)
                self.__LabelColorChoice.SetSelection(sel)

        self.__label_color = color
        for p in self.__sensor_plots:
            p.set_label_color(color)
        self.draw()

    def set_label_text(self, text='name'):
        """Add/remove sensor labels

        Parameters
        ----------
        labels : None | 'name' | 'index'
            Content of the labels. For 'name', any prefix common to all names
            is removed; with 'fullname', the full name is shown.
        """
        self.__check_label_arg(text)
        if hasattr(self, '_SensorLabelChoice'):
            sel = self.__label_option_args.index(text)
            self.__LabelChoice.SetSelection(sel)

        for p in self.__sensor_plots:
            p.show_labels(text, color=self.__label_color)
        self.draw()


class SensorMaps(_EelFigure):
    """Multiple views on a sensor layout.

    Allows selecting sensor groups and retrieving corresponding indices.

    Parameters
    ----------
    sensors : Sensor | NDVar
        The sensors to use, or an NDVar with a sensor dimension.
    select : list of int
        Initial selection.
    proj : str
        Sensor projection for the fourth plot.
    frame : scalar
        Size of the empty space around sensors in axes.
    title : None | string
        Figure title.

    Notes
    -----
    **Selecting Sensor Groups:**

     - Dragging with the left mouse button adds sensors to the selection.
     - Dragging with the right mouse button removes sensors from the current
       selection.
     - The 'Clear' button (or :meth:`clear`) clears the selection.

    """
    def __init__(self, sensors, select=[], proj='default', frame=0.05,
                 *args, **kwargs):
        sensors = as_sensor(sensors)

        # layout figure
        ftitle = 'SensorMaps'
        sens_name = getattr(sensors, 'sysname', None)
        if sens_name:
            ftitle = '%s: %s' % (ftitle, sens_name)

        self._drag_ax = None
        self._drag_x = None
        self._drag_y = None
        _EelFigure.__init__(self, ftitle, 4, 3, 1, False, ncol=2, nrow=2, *args,
                            **kwargs)
        self.figure.subplots_adjust(left=0, bottom=0, right=1, top=1,
                                    wspace=.1, hspace=.1)

        # store args
        self._sensors = sensors

        ext = np.vstack((sensors.locs.min(0), sensors.locs.max(0)))
        aframe = np.array([-frame, frame])
        xlim = ext[:, 0] + aframe
        ylim = ext[:, 1] + aframe
        zlim = ext[:, 2] + aframe

        # back
        ax = self.ax0 = self.figure.add_subplot(2, 2, 1)
        ax.proj = 'y-'
        ax.extent = False
        self._h0 = _ax_map2d(ax, sensors, proj=ax.proj, extent=ax.extent)

        # left
        ax = self.ax1 = self.figure.add_subplot(2, 2, 2, sharey=self.ax0)
        ax.proj = 'x-'
        ax.extent = False
        self._h1 = _ax_map2d(ax, sensors, proj=ax.proj, extent=ax.extent)

        # top
        ax = self.ax2 = self.figure.add_subplot(2, 2, 3, sharex=self.ax0)
        ax.proj = 'z+'
        ax.extent = False
        self._h2 = _ax_map2d(ax, sensors, proj=ax.proj, extent=ax.extent)

        self.ax0.set_xlim(*xlim)
        self.ax0.set_ylim(*zlim)
        self.ax1.set_xlim(*zlim)
        self.ax2.set_ylim(*ylim)

        # proj
        ax = self.ax3 = self.figure.add_subplot(2, 2, 4)
        ax.proj = proj
        ax.extent = 1
        self._h3 = _ax_map2d(ax, sensors, proj=ax.proj, extent=ax.extent)
        self.ax3.set_xlim(-frame, 1 + frame)
        self.ax3.set_ylim(-frame, 1 + frame)

        self._sensor_maps = (self._h0, self._h1, self._h2, self._h3)
        self._show()

        # selection
        self.sel_kwargs = dict(marker='o', s=5, c='r', linewidths=.9)
        self._sel_h = []
        if select is not None:
            self.set_selection(select)
        else:
            self.select = None

        # setup mpl event handling
        self.canvas.mpl_connect("button_press_event", self._on_button_press)
        self.canvas.mpl_connect("button_release_event", self._on_button_release)

    def _fill_toolbar(self, tb):
        import wx

        tb.AddSeparator()

        # plot labels
        btn = wx.Button(tb, wx.ID_CLEAR, "Clear")
        tb.AddControl(btn)
        btn.Bind(wx.EVT_BUTTON, self._OnClear)

    def clear(self):
        "Clear the current sensor selection."
        self.select = None
        self.update_mark_plot()

    def get_selection(self):
        """
        Returns
        -------
        selection : list
            Returns the current selection as a list of indices.
        """
        if self.select is None:
            return []
        else:
            return np.where(self.select)[0]

    def _on_button_press(self, event):
        ax = event.inaxes
        if ax:
            self._is_dragging = True
            self._drag_ax = event.inaxes
            self._drag_x = event.xdata
            self._drag_y = event.ydata

            self.canvas.store_canvas()
            x = np.ones(5) * event.xdata
            y = np.ones(5) * event.ydata
            self._drag_rect = ax.plot(x, y, '-k')[0]

    def _on_button_release(self, event):
        if not hasattr(self, '_drag_rect'):
            return

        x = self._drag_rect.get_xdata()
        y = self._drag_rect.get_ydata()
        xmin = min(x)
        xmax = max(x)
        ymin = min(y)
        ymax = max(y)

        ax = self._drag_ax
        locs = self._sensors.get_locs_2d(ax.proj, extent=ax.extent)
        x = locs[:, 0]
        y = locs[:, 1]
        sel = (x > xmin) & (x < xmax) & (y > ymin) & (y < ymax)

        if self.select is None:
            self.select = sel
        elif event.button == 1:
            self.select[sel] = True
        else:
            self.select[sel] = False

        # clear dragging-related attributes
        self._drag_rect.remove()
        del self._drag_rect
        self._drag_ax = None
        self._drag_x = None
        self._drag_y = None

        self.update_mark_plot()

    def _on_motion(self, event):
        super(self.__class__, self)._on_motion(event)
        ax = event.inaxes
        if ax and ax is self._drag_ax:
            x0 = self._drag_x
            x1 = event.xdata
            y0 = self._drag_y
            y1 = event.ydata
            x = [x0, x1, x1, x0, x0]
            y = [y0, y0, y1, y1, y0]
            self._drag_rect.set_data(x, y)
            self.canvas.redraw(artists=[self._drag_rect])

    def _OnClear(self, event):
        self.clear()

    def set_selection(self, select):
        """
        Set the current selection with a list of indices.

        Parameters
        ----------
        select : sensor index
            Index for sensor dimension, for example array_like of int, or list
            of sensor names.
        """
        idx = self._sensors.dimindex(select)
        self.select = np.zeros(len(self._sensors), dtype=bool)
        self.select[idx] = True
        self.update_mark_plot()

    def update_mark_plot(self):
        for h in self._sensor_maps:
            h.sensors.mark_sensors(self.select, **self.sel_kwargs)
        self.canvas.draw()


class SensorMap(SensorMapMixin, _EelFigure):
    """Plot sensor positions in 2 dimensions

    Parameters
    ----------
    sensors : NDVar | Sensor
        sensor-net object or object containing sensor-net
    labels : None | 'index' | 'name' | 'fullname'
        Content of the labels. For 'name', any prefix common to all names
        is removed; with 'fullname', the full name is shown.
    proj:
        Transform to apply to 3 dimensional sensor coordinates for plotting
        locations in a plane
    mark : None | list of int
        List of sensor indices to mark.
    frame : scalar
        Size of the empty space around sensors in axes.
    connectivity : bool
        Show sensor connectivity (default False).
    title : None | string
        Figure title.
    """
    def __init__(self, sensors, labels='name', proj='default', mark=None,
                 frame=.05, connectivity=False, *args, **kwargs):
        sensors = as_sensor(sensors)

        if sensors.sysname:
            ftitle = 'SensorMap: %s' % sensors.sysname
        else:
            ftitle = 'SensorMap'
        _EelFigure.__init__(self, ftitle, 1, 7, 1, False, *args, **kwargs)
        self.axes = self._axes[0]

        # store args
        self._sensors = sensors
        self._proj = proj
        self._marker_handles = []
        self._connectivity = None

        self._markers = _ax_map2d(self.axes, sensors, proj, frame=frame)
        SensorMapMixin.__init__(self, [self._markers.sensors])

        if labels:
            self.set_label_text(labels)
        if mark is not None:
            self.mark_sensors(mark)

        if connectivity:
            self.show_connectivity()

        self._show()

    def mark_sensors(self, mark, kwargs=dict(marker='o',  # symbol
                                             color='r',  # mpl plot kwargs ...
                                             ms=5,  # marker size
                                             markeredgewidth=.9,
                                             ls='',
                                             )):
        """Mark specific sensors.

        Parameters
        ----------
        mark : sequence of {str | int}
            List of sensor names or indices.
        kwargs : dict
            Dict with kwargs for customizing the sensor plot (matplotlib plot
            kwargs).

        See Also
        --------
        .remove_markers() : Remove the markers
        """
        h = _plt_map2d(self.axes, self._sensors, self._proj, mark=mark,
                       kwargs=kwargs)
        self._marker_handles.append(h)
        self.canvas.draw()

    def remove_markers(self):
        "Remove all sensor markers."
        while len(self._marker_handles) > 0:
            h = self._marker_handles.pop(0)
            h.remove()
        self.canvas.draw()

    def show_connectivity(self, show=True):
        """Show the sensor connectivity as lines connecting sensors.

        Parameters
        ----------
        show : None | True | scalar
            True to show the default connectivity.
            None to remove the connectivity lines.
            Scalar to plot connectivity for a different connect_dist parameter
            (see Sensor.connectivity()).
        """
        if not show:
            self._markers.connectivity.show(None)
        else:
            if show is True:
                conn = self._sensors.connectivity()
            else:
                conn = self._sensors.connectivity(show)
            self._markers.connectivity.show(conn)
        self.draw()


def map3d(sensors, marker='c*', labels=False, head=0):
    """3d plot of a Sensors instance"""
    import matplotlib.pyplot as plt

    sensors = as_sensor(sensors)

    locs = sensors.locs
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(locs[:, 0], locs[:, 1], locs[:, 2])
    # plot head ball
    if head:
        u = np.linspace(0, 1 * np.pi, 10)
        v = np.linspace(0, np.pi, 10)

        x = 5 * head * np.outer(np.cos(u), np.sin(v))
        z = 10 * (head * np.outer(np.sin(u), np.sin(v)) - .5)  # vertical
        y = 5 * head * np.outer(np.ones(np.size(u)), np.cos(v))  # axis of the sphere
        ax.plot_surface(x, y, z, rstride=1, cstride=1, color='w')
    # n = 100
    # for c, zl, zh in [('r', -50, -25), ('b', -30, -5)]:
    # xs, ys, zs = zip(*
    #               [(random.randrange(23, 32),
    #                 random.randrange(100),
    #                 random.randrange(zl, zh)
    #                 ) for i in range(n)])
    # ax.scatter(xs, ys, zs, c=c)
