'''
Plot sensor maps.
'''
# author: Christian Brodbeck


import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from .._data_obj import Datalist
from . import _base


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
            line = plt.Line2D(x, y, **linestyle)
            self.ax.add_line(line)
            self._h.append(line)


def _ax_map2d_fast(ax, sensors, proj='default',
                   m='x', mew=.5, mc='b', ms=3,):
    locs = sensors.get_locs_2d(proj=proj)
    h = plt.plot(locs[:, 0], locs[:, 1], m, color=mc, ms=ms, markeredgewidth=mew)

    return h


class _ax_map2d:
    def __init__(self, ax, sensors, proj='default', extent=1,
                 frame=.02,
                 kwargs=dict(
                             marker='x',  # symbol
                             color='b',  # mpl plot kwargs ...
                             ms=3,  # marker size
                             markeredgewidth=.5,
                             ls='',
                             ),
                 ):
        self.ax = ax

        ax.set_aspect('equal')
        ax.set_frame_on(False)
        ax.set_axis_off()

        h = _plt_map2d(ax, sensors, proj=proj, extent=extent, kwargs=kwargs)
        self.sensors = h

        locs = sensors.get_locs_2d(proj=proj, extent=extent)
        self.connectivity = _plt_connectivity(ax, locs, None)

        ax.set_xlim(-frame, 1 + frame)

    def remove(self):
        "remove from axes"
        self.sensors.remove()


class _plt_map2d:
    def __init__(self, ax, sensors, proj='default', extent=1, mark=None,
                 kwargs=dict(
                             marker='.',  # symbol
                             color='k',  # mpl plot kwargs ...
                             ms=1,  # marker size
                             markeredgewidth=.5,
                             ls='',
                             ),
                 ):
        """
        Parameters
        ----------
        ax : matplotlib Axes
            Axes.
        sensors : Sensor
            Sensor dimension.
        """
        self.ax = ax
        self.sensors = sensors
        self.locs = sensors.get_locs_2d(proj=proj, extent=extent)
        self._mark_handles = None

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
            Sensors which should be marked
        marker : str
            Matplotlib marker specification for the marked sensors.
        """
        while self._mark_handles:
            self._mark_handles.pop().remove()

        if not np.count_nonzero(sensors):
            return

        idx = self.sensors.dimindex(sensors)
        locs = self.locs[idx]
        self._mark_handles = self.ax.plot(locs[:, 0], locs[:, 1], *args,
                                          **kwargs)

    def remove(self):
        "remove from axes"
        while self._mark_handles:
            self._mark_handles.pop().remove()
        while self.markers:
            self.markers.pop().remove()

    def show_labels(self, text='idx', xpos=0, ypos=.01, **text_kwargs):
        """Plot labels for the sensors

        Parameters
        ----------
        text : None | 'idx' | 'name' | 'fullname'
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
        if text == 'idx':
            labels = map(str, xrange(len(sensors)))
        elif text == 'name':
            labels = sensors.names
            prefix = os.path.commonprefix(labels)
            pf_len = len(prefix)
            labels = [label[pf_len:] for label in labels]
        elif text == 'fullname':
            labels = sensors.names
        else:
            err = "text has to be 'idx' or 'name', can't be %r" % text
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


class _tb_sensors_mixin:
    # expects self._sensor_plots to be list of _plt_map2d
    def __init__(self):
        self._label_color = 'k'

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

    def _OnSensorLabelChoice(self, event):
        sel = event.GetSelection()
        text = [None, 'idx', 'name'][sel]
        self.set_label_text(text)

    def _OnSensorLabelColorChoice(self, event):
        sel = event.GetSelection()
        color = ['k', 'w', 'b', 'g', 'r', 'c', 'm', 'y'][sel]
        self.set_label_color(color)

    def mark_sensors(self, sensors, marker='bo'):
        for p in self._sensor_plots:
            p.mark_sensors(sensors, marker)
        self.draw()

    def set_label_color(self, color='w'):
        if hasattr(self, '_SensorLabelChoice'):
            sels = ['k', 'w', 'b', 'g', 'r', 'c', 'm', 'y']
            if color in sels:
                sel = sels.index(color)
                self._SensorLabelColorChoice.SetSelection(sel)

        self._label_color = color
        for p in self._sensor_plots:
            p.set_label_color(color)
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

        for p in self._sensor_plots:
            p.show_labels(text, color=self._label_color)
        self.draw()


class SensorMaps(_base._EelFigure):
    """
    GUI with multiple views on a sensor layout.

    Allows selecting sensor groups and retrieving corresponding indices.


    **Selecting Sensor Groups:**

     - Dragging with the left mouse button adds sensors to the selection.
     - Dragging with the right mouse button removes sensors from the current
       selection.
     - The 'Clear' button (or :meth:`clear`) clears the selection.

    """
    def __init__(self, sensors, select=[], proj='default', frame=0.05,
                 title=None, **layout):
        """
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
        """
        # in case Sensors parent is submitted
        if hasattr(sensors, 'sensors'):
            sensors = sensors.sensors
        elif hasattr(sensors, 'sensor'):
            sensors = sensors.sensor

        # layout figure
        layout.update(ncol=2, nrow=2)
        ftitle = 'SensorMaps'
        sens_name = getattr(sensors, 'sysname', None)
        if sens_name:
            ftitle = '%s: %s' % (ftitle, sens_name)

        self._drag_ax = None
        self._drag_x = None
        self._drag_y = None
        super(SensorMaps, self).__init__(ftitle, 4, layout, 1, 3, figtitle=title)
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
        ax.y_fmt = 'z = %.3g'
        ax.x_fmt = 'x = %.3g'
        ax.proj = 'y-'
        ax.extent = False
        self._h0 = _ax_map2d(ax, sensors, proj=ax.proj, extent=ax.extent)

        # left
        ax = self.ax1 = self.figure.add_subplot(2, 2, 2, sharey=self.ax0)
        ax.x_fmt = 'y = %.3g'
        ax.y_fmt = 'z = %.3g'
        ax.proj = 'x-'
        ax.extent = False
        self._h1 = _ax_map2d(ax, sensors, proj=ax.proj, extent=ax.extent)

        # top
        ax = self.ax2 = self.figure.add_subplot(2, 2, 3, sharex=self.ax0)
        ax.x_fmt = 'x = %.3g'
        ax.y_fmt = 'y = %.3g'
        ax.proj = 'z+'
        ax.extent = False
        self._h2 = _ax_map2d(ax, sensors, proj=ax.proj, extent=ax.extent)

        self.ax0.set_xlim(*xlim)
        self.ax0.set_ylim(*zlim)
        self.ax1.set_xlim(*zlim)
        self.ax2.set_ylim(*ylim)

        # proj
        ax = self.ax3 = self.figure.add_subplot(2, 2, 4)
        ax.x_fmt = ' '
        ax.y_fmt = ' '
        ax.proj = proj
        ax.extent = 1
        self._h3 = _ax_map2d(ax, sensors, proj=ax.proj, extent=ax.extent)
        self.ax3.set_xlim(-frame, 1 + frame)
        self.ax3.set_ylim(-frame, 1 + frame)

        self._sensor_maps = (self._h0, self._h1, self._h2, self._h3)
        self._show(tight=False)

        # selection
        self.sel_kwargs = dict(marker='o',  # symbol
                               color='r',  # mpl plot kwargs ...
                               ms=5,  # marker size
                               markeredgewidth=.9,
                               ls='')
        self._sel_h = []
        if select is not None:
            self.set_selection(select)
        else:
            self.select = None

        # setup mpl event handling
        self.canvas.mpl_connect("button_press_event", self._on_button_press)
        self.canvas.mpl_connect("button_release_event", self._on_button_release)

    def _fill_toolbar(self, tb):
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
        select : list of int
            List of sensor indices in the new selection.
        """
        idx = self._sensors.dimindex(select)
        self.select = np.zeros(len(self._sensors), dtype=bool)
        self.select[idx] = True
        self.update_mark_plot()

    def update_mark_plot(self):
        for h in self._sensor_maps:
            h.sensors.mark_sensors(self.select, **self.sel_kwargs)
        self.canvas.draw()





class SensorMap2d(_tb_sensors_mixin, _base._EelFigure):
    """
    Plot a 2d Sensor Map.

    """
    def __init__(self, sensors, labels='name', proj='default', mark=None,
                 frame=.05, title=None, **layout):
        """Plot sensor positions in 2 dimensions

        Parameters
        ----------
        sensors : NDVar | Sensor
            sensor-net object or object containing sensor-net
        labels : None | 'idx' | 'name' | 'fullname'
            Content of the labels. For 'name', any prefix common to all names
            is removed; with 'fullname', the full name is shown.
        proj:
            Transform to apply to 3 dimensional sensor coordinates for plotting
            locations in a plane
        mark : None | list of int
            List of sensor indices to mark.
        frame : scalar
            Size of the empty space around sensors in axes.
        title : None | string
            Figure title.
        """
        # in case Sensors parent is submitted
        if hasattr(sensors, 'sensors'):
            sensors = sensors.sensors
        elif hasattr(sensors, 'sensor'):
            sensors = sensors.sensor

        ftitle = 'SensorMap2d'
        sens_name = getattr(sensors, 'sysname', None)
        if sens_name:
            ftitle = '%s: %s' % (ftitle, sens_name)
        _base._EelFigure.__init__(self, ftitle, 1, layout, 1, 7, figtitle=title)
        _tb_sensors_mixin.__init__(self)

        # store args
        self._sensors = sensors
        self._proj = proj
        self._marker_handles = []
        self._connectivity = None

        ax = self.figure.add_axes([frame, frame, 1 - 2 * frame, 1 - 2 * frame])
        self.axes = ax
        self._markers = _ax_map2d(ax, sensors, proj=proj)
        self._sensor_plots = [self._markers.sensors]
        if labels:
            self.set_label_text(labels)
        if mark is not None:
            self.mark_sensors(mark)

        self._show(tight=False)

    def mark_sensors(self, mark, kwargs=dict(marker='o',  # symbol
                                             color='r',  # mpl plot kwargs ...
                                             ms=5,  # marker size
                                             markeredgewidth=.9,
                                             ls='',
                                             )):
        """Mark specific sensors.

        Parameters
        ----------
        mark : list of int
            List of sensor indices.
        kwargs : dict
            Dict with kwargs for customizing the sensor plot (matplotlib plot
            kwargs).

        See Also
        --------
        .remove_markers() : Remove the markers
        """
        h = _plt_map2d(self.axes, self._sensors, proj=self._proj, mark=mark,
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
    if hasattr(sensors, 'sensors'):
        sensors = sensors.sensors
    elif hasattr(sensors, 'sensor'):
        sensors = sensors.sensor

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
