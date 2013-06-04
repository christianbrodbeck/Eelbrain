'''
plot.sensors
============

Plotting functions for Sensor dimension objects.

'''
# author: Christian Brodbeck


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from ..vessels.data import datalist

try:
    import wx
    _ID_label_Ids = wx.NewId()
    _ID_label_names = wx.NewId()
    _ID_label_None = wx.NewId()
except:
    pass

import _base



__hide__ = ['mpl_canvas']



# some useful kwarg dictionaries for different plot layouts
kwargs_mono = dict(mc='k',
                   lc='.5',
                   hllc='k',
                   hlmc='k',
                   hlms=7,
                   strlc='k')


def _ax_map2d_fast(ax, sensors, proj='default',
                   m='x', mew=.5, mc='b', ms=3,):
    locs = sensors.get_locs_2d(proj=proj)
    h = plt.plot(locs[:, 0], locs[:, 1], m, color=mc, ms=ms, markeredgewidth=mew)

    return h


def _ax_map2d(ax, sensors, proj='default', extent=1,
              frame=.02,
              kwargs=dict(
                          marker='x',  # symbol
                          color='b',  # mpl plot kwargs ...
                          ms=3,  # marker size
                          markeredgewidth=.5,
                          ls='',
                          ),
              ):
    ax.set_aspect('equal')
    ax.set_frame_on(False)
    ax.set_axis_off()

    h = _plt_map2d(ax, sensors, proj=proj, extent=extent, kwargs=kwargs)

    ax.set_xlim(-frame, 1 + frame)
    return h



class _plt_map2d:
    def __init__(self, ax, sensors, proj='default', extent=1, ROI=None,
                 kwargs=dict(
                             marker='.',  # symbol
                             color='k',  # mpl plot kwargs ...
                             ms=1,  # marker size
                             markeredgewidth=.5,
                             ls='',
                             ),
                 ):
        self.ax = ax
        self.sensors = sensors
        locs = sensors.get_locs_2d(proj=proj, extent=extent)
        self.ROI = ROI
        if ROI is not None:
            locs = locs[ROI]
        self.locs = locs

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

    def show_labels(self, text='idx', xpos=0, ypos=.01, **text_kwargs):
        """Plot labels for the sensors

        Parameters
        ----------
        text : None | 'idx' | 'name'
            Kind of label: sensor index
        xpos, ypos : scalar
            The position offset of the labels from the sensor markers.
        text_kwargs : **
            Matplotlib text parameters.
        """
        # remove existing labels
        while self.labels:
            h = self.labels.pop()
            h.remove()

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
        else:
            err = "text has to be 'idx' or 'name', can't be %r" % text
            raise NotImplementedError(err)

        if self.ROI is not None:
            labels = datalist(labels)[self.ROI]

        locs = self.locs + [[xpos, ypos]]
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



class multi(_base.eelfigure):
    """
    GUI with multiple views on a sensor layout.

    Allows selecting sensor groups (ROIs) and retrieving corresponding indices.


    **Selecting ROIs:**

     - Dragging with the left mouse button adds sensors to the ROI.
     - Dragging with the right mouse button removes sensors from the current
       ROI.
     - The 'Clear' button (or :meth:`clear`) removes the ROI.

    """
    def __init__(self, sensors, size=7, dpi=100, frame=.05, ROI=[], proj='default'):
        """
        Parameters
        ----------
        sensors : Sensor | ndvar
            The sensors to use, or an ndvar with a sensor dimension.
        ROI : list of int
            Initial ROI.
        proj : str
            Sensor projection for the fourth plot.

        """
        title = "Sensors: %s" % getattr(sensors, 'sysname', '')
        super(multi, self).__init__(title=title, figsize=(size, size), dpi=dpi)

        # in case Sensors parent is submitted
        if hasattr(sensors, 'sensors'):
            sensors = sensors.sensors
        elif hasattr(sensors, 'sensor'):
            sensors = sensors.sensor

        # store args
        self._sensors = sensors

        self.figure.set_facecolor('w')
        self.figure.subplots_adjust(left=0, bottom=0, right=1, top=1,
                                    wspace=.1, hspace=.1)

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

        self.axes = [self.ax0, self.ax1, self.ax2, self.ax3]
        self._show()

        # ROI
        self.ROI_kwargs = dict(marker='o',  # symbol
                               color='r',  # mpl plot kwargs ...
                               ms=5,  # marker size
                               markeredgewidth=.9,
                               ls='')
        self._ROI_h = []
        if ROI:
            self.set_ROI(ROI)
        else:
            self.ROI = None

        # setup mpl event handling
        self.canvas.mpl_connect("button_press_event", self._on_button_press)
        self.canvas.mpl_connect("button_release_event", self._on_button_release)
        self._drag_ax = None
        self._drag_x = None
        self._drag_y = None

    def _fill_toolbar(self, tb):
        tb.AddSeparator()

        # plot labels
        btn = wx.Button(tb, wx.ID_CLEAR, "Clear")
        tb.AddControl(btn)
        btn.Bind(wx.EVT_BUTTON, self._OnClear)

    def clear(self):
        "Clear the current ROI selection."
        self.ROI = None
        self.update_ROI_plot()

    def get_ROI(self):
        """
        Returns
        -------
        ROI : list
            Returns the current ROI as a list of indices.
        """
        if self.ROI is None:
            return []
        else:
            return np.where(self.ROI)[0]

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

        if self.ROI is None:
            self.ROI = sel
        elif event.button == 1:
            self.ROI[sel] = True
        else:
            self.ROI[sel] = False

        # clear dragging-related attributes
        self._drag_rect.remove()
        del self._drag_rect
        self._drag_ax = None
        self._drag_x = None
        self._drag_y = None

        self.update_ROI_plot()

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

    def set_ROI(self, ROI):
        """
        Set the current ROI with a list of indices.

        Parameters
        ----------
        ROI : list of int
            List of sensor indices in the new ROI.
        """
        self.ROI = np.zeros(len(self._sensors), dtype=bool)
        self.ROI[ROI] = True
        self.update_ROI_plot()

    def update_ROI_plot(self):
        # remove old plots
        while self._ROI_h:
            self._ROI_h.pop(0).remove()

        # plot
        if self.ROI is not None:
            for ax in self.axes:
                h = _plt_map2d(ax, self._sensors, proj=ax.proj, extent=ax.extent,
                               ROI=self.ROI, kwargs=self.ROI_kwargs)
                self._ROI_h.extend(h)

        self.canvas.draw()





class map2d(_base.eelfigure):
    """
    Plot a 2d Sensor Map.

    """
    def __init__(self, sensors, labels='idx', proj='default', ROI=None,
                 size=6, dpi=100, frame=.05, **kwargs):
        """Plot sensor positions in 2 dimensions

        Parameters
        ----------
        sensors : ndvar | Sensor
            sensor-net object or object containing sensor-net
        labels : 'idx' | 'name'
            how the sensors should be labelled
        proj:
            Transform to apply to 3 dimensional sensor coordinates for plotting
            locations in a plane
        """
        title = "Sensors: %s" % getattr(sensors, 'sysname', '')
        super(map2d, self).__init__(title=title, figsize=(size, size), dpi=dpi)

        # in case Sensors parent is submitted
        if hasattr(sensors, 'sensors'):
            sensors = sensors.sensors
        elif hasattr(sensors, 'sensor'):
            sensors = sensors.sensor

        # store args
        self._sensors = sensors
        self._proj = proj
        self._ROIs = []

        self.figure.set_facecolor('w')
        ax = self.figure.add_axes([frame, frame, 1 - 2 * frame, 1 - 2 * frame])
        self.axes = ax
        self._markers = _ax_map2d(ax, sensors, proj=proj, **kwargs)
        if labels:
            self._markers.show_labels(labels)
        if ROI is not None:
            self.plot_ROI(ROI)

        self._show()

    def _fill_toolbar(self, tb):
        tb.AddSeparator()

        # plot labels
        for Id, name in [(_ID_label_None, "No Labels"),
                         (_ID_label_Ids, "Indexes"),
                         (_ID_label_names, "Names"), ]:
            btn = wx.Button(tb, Id, name)
            tb.AddControl(btn)
            btn.Bind(wx.EVT_BUTTON, self._OnPlotLabels)

    def _OnPlotLabels(self, event):
        Id = event.GetId()
        labels = {_ID_label_None: None,
                  _ID_label_Ids: "idx",
                  _ID_label_names: "name"}[Id]
        self.plot_labels(labels)

    def plot_labels(self, labels='idx'):
        """
        Add labels to all sensors.

        Parameters
        ----------
        labels : None | 'idx' | 'name'
            Content of the labels.
        """
        self._markers.show_labels(labels)
        self.canvas.draw()

    def plot_ROI(self, ROI, kwargs=dict(marker='o',  # symbol
                                        color='r',  # mpl plot kwargs ...
                                        ms=5,  # marker size
                                        markeredgewidth=.9,
                                        ls='',
                                        )):
        """
        Mark sensors in a ROI.

        Parameters
        ----------
        ROI : list of int
            List of sensor indices.
        kwargs : dict
            Dict with kwargs for customizing the sensor plot (matplotlib plot
            kwargs).
        """
        h = _plt_map2d(self.axes, self._sensors, proj=self._proj, ROI=ROI, kwargs=kwargs)
        self._ROIs.extend(h)
        self.canvas.draw()

    def remove_ROIs(self):
        "Remove all marked ROIs."
        while len(self._ROIs) > 0:
            h = self._ROIs.pop(0)
            h.remove()
        self.canvas.draw()



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
