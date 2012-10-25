'''
Plotting functions for sensor_net instances.


@author: christianmbrodbeck
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    import wx
    _ID_label_Ids = wx.NewId()
    _ID_label_names = wx.NewId()
    _ID_label_None = wx.NewId()
except:
    pass

try:
    from _sensors_mayavi import coreg, mrk_fix
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


def _ax_map2d_fast(ax, sensor_net, proj='default',
                   m='x', mew=.5, mc='b', ms=3,):
    locs = sensor_net.getLocs2d(proj=proj)
    h = plt.plot(locs[:, 0], locs[:, 1], m, color=mc, ms=ms, markeredgewidth=mew)

    return h


def _ax_map2d(ax, sensor_net, proj='default', extent=1,
              frame=.02,
              kwargs=dict(
                          marker='x', # symbol
                          color='b', # mpl plot kwargs ...
                          ms=3, # marker size
                          markeredgewidth=.5,
                          ls='',
                          ),
              ):
    ax.set_aspect('equal')
    ax.set_frame_on(False)
    ax.set_axis_off()

    h = _plt_map2d(ax, sensor_net, proj=proj, extent=extent, kwargs=kwargs)

    ax.set_xlim(-frame, 1 + frame)
    return h



def _plt_map2d(ax, sensor_net, proj='default', extent=1, ROI=None,
               kwargs=dict(
                           marker='.', # symbol
                           color='k', # mpl plot kwargs ...
                           ms=1, # marker size
                           markeredgewidth=.5,
                           ls='',
                           ),
               ):
    locs = sensor_net.getLocs2d(proj=proj, extent=extent)
    if ROI is not None:
        locs = locs[ROI]

    if 'color' in kwargs:
        h = ax.plot(locs[:, 0], locs[:, 1], **kwargs)
    else:
        h = []
        colors = mpl.rcParams['axes.color_cycle']
        nc = len(colors)
        for i in xrange(len(locs)):
            kwargs['color'] = kwargs['mec'] = colors[i % nc]
            hi = ax.plot(locs[i, 0], locs[i, 1], **kwargs)
            h.append(hi)
    return h



def _plt_map2d_labels(ax, sensor_net, proj='default',
                      text='id', # 'id', 'name'
                      xpos=0, # horizontal distance from marker
                      ypos=.01, # vertical distance from marker
                      kwargs=dict(# mpl text kwargs ...
                                  color='k',
                                  fontsize=8,
                                  horizontalalignment='center',
                                  verticalalignment='bottom',
                                  ),
                      ):
    if text == 'id':
        labels = map(str, xrange(len(sensor_net)))
    elif text == 'name':
        labels = sensor_net.names
    else:
        err = "text has to be 'id' or 'name', can't be %r" % text
        raise NotImplementedError(err)

    locs = sensor_net.getLocs2d(proj=proj)

    handles = []
    for i in xrange(len(labels)):
        x = locs[i, 0] + xpos
        y = locs[i, 1] + ypos
        lbl = labels[i]
        h = ax.text(x, y, lbl, **kwargs)
        handles.append(h)

    return handles



class multi(_base.eelfigure):
    """
    Select ROIs
    -----------

     - Dragging with the left mouse button adds sensors to the ROI.
     - Dragging with the right mouse button removes sensors from the current ROI.
     - The 'Clear' button (or self.clear()) removes the ROI.


    Methods
    -------

    clear :
        Clear the current ROI selection.
    get_ROI : list
        Returns the current ROI as a list of indices.
    set_ROI
        Set the current ROI with a list of indices.

    """
    def __init__(self, sensors, size=7, dpi=100, frame=.05, ROI=[], proj='default'):
        title = "Sensor Net: %s" % getattr(sensors, 'name', '')
        super(multi, self).__init__(title=title, figsize=(size, size), dpi=dpi)

        # in case sensor_net parent is submitted
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
        self.ROI_kwargs = dict(marker='o', # symbol
                               color='r', # mpl plot kwargs ...
                               ms=5, # marker size
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
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
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
        "clear the ROI"
        self.ROI = None
        self.update_ROI_plot()

    def get_ROI(self):
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
        locs = self._sensors.getLocs2d(ax.proj, extent=ax.extent)
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
    2d Sensor Map, Methods:

    plot_ROI:
        mark sensors in a ROI
    plot_labels:
        add labels to the sensors
    remove_ROIs:
        remove all marked ROIs

    """
    def __init__(self, sensor_net, labels='id', proj='default', ROI=None,
                 size=6, dpi=100, frame=.05, **kwargs):
        """
        **Parameters:**

        sensor_net :
            sensor-net object or object containing sensor-net

        labels : 'id' | 'name'
            how the sensors should be labelled

        proj:
            Transform to apply to 3 dimensional sensor coordinates for plotting
            locations in a plane

        """
        title = "Sensor Net: %s" % getattr(sensor_net, 'name', '')
        super(map2d, self).__init__(title=title, figsize=(size, size), dpi=dpi)

        # in case sensor_net parent is submitted
        if hasattr(sensor_net, 'sensors'):
            sensor_net = sensor_net.sensors
        elif hasattr(sensor_net, 'sensor'):
            sensor_net = sensor_net.sensor

        # store args
        self._sensor_net = sensor_net
        self._proj = proj
        self._ROIs = []

        self.figure.set_facecolor('w')
        ax = self.figure.add_axes([frame, frame, 1 - 2 * frame, 1 - 2 * frame])
        self.axes = ax
        self._marker_h = _ax_map2d(ax, sensor_net, proj=proj, **kwargs)
        if ROI is not None:
            self.plot_ROI(ROI)

        self._label_h = None
        if labels:
            self.plot_labels(labels=labels)

        self._show()

    def _fill_toolbar(self, tb):
        tb.AddSeparator()

        # plot labels
        for Id, name in [(_ID_label_None, "No Labels"),
                         (_ID_label_Ids, "Ids"),
                         (_ID_label_names, "Names"), ]:
            btn = wx.Button(tb, Id, name)
            tb.AddControl(btn)
            btn.Bind(wx.EVT_BUTTON, self._OnPlotLabels)

    def _OnPlotLabels(self, event):
        Id = event.GetId()
        labels = {_ID_label_None: None,
                  _ID_label_Ids: "id",
                  _ID_label_names: "name"}[Id]
        self.plot_labels(labels)

    def plot_labels(self, labels='id'):
        """
        Add labels to all sensors. Possible values:

        'id':
            sensor indexes
        'name':
            sensor names

        """
        if self._label_h:
            for h in self._label_h:
                h.remove()

        if labels:
            h = _plt_map2d_labels(self.axes, self._sensor_net, proj=self._proj,
                                  text=labels)
        else:
            h = None
        self._label_h = h
        self.canvas.draw()

    def plot_ROI(self, ROI, kwargs=dict(marker='o', # symbol
                                        color='r', # mpl plot kwargs ...
                                        ms=5, # marker size
                                        markeredgewidth=.9,
                                        ls='',
                                        )):
        h = _plt_map2d(self.axes, self._sensor_net, proj=self._proj, ROI=ROI, kwargs=kwargs)
        self._ROIs.extend(h)
        self.canvas.draw()

    def remove_ROIs(self):
        while len(self._ROIs) > 0:
            h = self._ROIs.pop(0)
            h.remove()
        self.canvas.draw()





def map3d(sensor_net, marker='c*', labels=False, head=0):
    """3d plot of a sensor_net"""
    if hasattr(sensor_net, 'sensors'):
        sensor_net = sensor_net.sensors
    locs = sensor_net.locs
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(locs[:, 0], locs[:, 1], locs[:, 2])
    # plot head ball
    if head:
        u = np.linspace(0, 1 * np.pi, 10)
        v = np.linspace(0, np.pi, 10)

        x = 5 * head * np.outer(np.cos(u), np.sin(v))
        z = 10 * (head * np.outer(np.sin(u), np.sin(v)) - .5)         # vertical
        y = 5 * head * np.outer(np.ones(np.size(u)), np.cos(v))  # axis of the sphere
        ax.plot_surface(x, y, z, rstride=1, cstride=1, color='w')
    #n = 100
    #for c, zl, zh in [('r', -50, -25), ('b', -30, -5)]:
    #xs, ys, zs = zip(*
    #               [(random.randrange(23, 32),
    #                 random.randrange(100),
    #                 random.randrange(zl, zh)
    #                 ) for i in range(n)])
    #ax.scatter(xs, ys, zs, c=c)
