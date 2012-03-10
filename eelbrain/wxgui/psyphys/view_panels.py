'''
Created on Apr 13, 2011

@author: christian
'''

import logging

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure

import wx

from eelbrain.psyphys import visualizers

import ID



#####  Events  #####     #####     #####     #####     #####     #####     ####
myEVT_FIGURE_POS = wx.NewEventType()
EVT_FIGURE_POS = wx.PyEventBinder(myEVT_FIGURE_POS, 1)

class FigurePosEvent(wx.PyCommandEvent):
    """
    Event that is sent every time the user adjusts the view on a plot.
    
    """
    def __init__(self, t0, t1, id=-1):
        wx.PyCommandEvent.__init__(self, myEVT_FIGURE_POS, id)
        self.t0 = t0
        self.t1 = t1



#####  Frames to test individual panels  #####     #####     #####     #####
class OverviewFrame(wx.Frame):
    def __init__(self, parent, dataset):
        visualizer = visualizers.default(dataset)
        wx.Frame.__init__(self, parent)
        
        self.panel = OverviewPanel(self)        
        self.panel.plot_overview([visualizer])

class ZoomFrame(wx.Frame):
    def __init__(self, parent, dataset):
        visualizer = visualizers.default(dataset)
        wx.Frame.__init__(self, parent)
        
        self.panel = ZoomPanel(self)        
        self.panel.plot([visualizer])


#####  the Panels  #####     #####     #####     #####
class CanvasPanel(wx.Panel):
    def __init__(self, parent, id=wx.ID_ANY):
        wx.Panel.__init__(self, parent, id=id)
        
        self.SetBackgroundColour(wx.NamedColour("WHITE"))
        
        self.figure = Figure()
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.canvas.Bind(wx.EVT_ENTER_WINDOW, self.OnEnterWindow)
        
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.SetSizer(self.sizer)
        self.Fit()
        
        self.Bind(wx.EVT_PAINT, self.OnPaint)
    def OnEnterWindow(self, event):
        "http://matplotlib.sourceforge.net/examples/user_interfaces/wxcursor_demo.html"
        self.canvas.SetCursor(wx.StockCursor(wx.CURSOR_CROSS))
    def OnPaint(self, event):
        logging.debug('%s: OnPaint!!!' % self.__class__.__name__)
        self.canvas.draw()
    def SendFigurePosEvent(self, t0, t1):
        # issue event
        logging.debug("EVENT sent!!")
        evt = FigurePosEvent(t0, t1, id=self.GetId())
        self.GetEventHandler().ProcessEvent(evt)
    def OnFigurePosEvent(self, event):
#        logging.debug("EVENT received!!")
        self.zoom(event.t0, event.t1, send_event=False)




class ZoomPanel(CanvasPanel):
    def __init__(self, parent, id=ID.CANVAS_PANEL_ZOOM):
        CanvasPanel.__init__(self, parent, id=id)
        
    # User interaction
#        self._dragging = 0
#        self._dt = 10
        self.canvas.mpl_connect('motion_notify_event', self.OnCanvasMotion)
        self.canvas.mpl_connect('button_press_event', self.OnCanvasMouseDown)
        self.canvas.mpl_connect('button_release_event', self.OnCanvasMouseUp)
    
    def plot(self, visualizers):
        self.figure.clf()
        
        ax = self.ax = self.figure.add_axes([0, .1, 1, .9], frame_on=False)
        for v in visualizers:
            v.toax(ax, None, None)
        
        self._tmin = self._t0 = t0 = v.tstart
        self._tmax = self._t1 = t1 = v.tend
        ax.set_xlim(t0, t1)
    
    def relative_zoom(self, factor):
        t0 = self._t0
        t1 = self._t1
        diff = ((t1 - t0) * (factor - 1)) / 2.
        self.zoom(t0 + diff, t1 - diff)
    def relative_move(self, factor):
        t0 = self._t0
        t1 = self._t1
        diff = (t1 - t0) * factor
        self.zoom(t0 + diff, t1 + diff)
    
    def zoom(self, t0, t1, send_event=True):
        t0 = self._t0 = max(self._tmin, t0)
        t1 = self._t1 = min(self._tmax, t1)
        self.ax.set_xlim(t0, t1)
        self.canvas.draw()
        
        if send_event:
            self.SendFigurePosEvent(t0, t1)
        
    def move(self, dt):
        t0 = self._t0 + dt
        t1 = self._t1 + dt
        if t0 < self._tmin:
            t1 += self._tmin - t0
            t0 = self._tmin
        elif t1 > self._tmax:
            t0 -= t1 - self._tmax
            t1 = self._tmax
        self.zoom(t0, t1)
    
    def move_t_to_x(self, t, x, send_event=True):
        # event.x is position within renderer
        width = self.canvas.get_renderer().width
        x_ratio = x / width
        dt = self._t1 - self._t0
#        t_rel = t - self._t0
#        t_ratio = t_rel / dt
        t0 = t - x_ratio * dt
        t1 = t0 + dt
        self.zoom(t0, t1, send_event=send_event)
        
    def OnCanvasMouseDown(self, event):
        if event.inaxes:
            self._t_grip = event.xdata
            self._xlim_grip = self._t0, self._t1
            self._grip_dt = 0
#            self._x_grip = event.x
#            self._r_width = self.canvas.get_renderer().width
            if wx.__version__ >= '2.9':
                CursorId = wx.CURSOR_CLOSED_HAND
            else:
                CursorId = wx.CURSOR_BULLSEYE
            self.canvas.SetCursor(wx.StockCursor(CursorId))
    def OnCanvasMouseUp(self, event):
        self.canvas.SetCursor(wx.StockCursor(wx.CURSOR_CROSS))
    def OnCanvasMotion(self, event):
#        logging.debug('x=%.2f, xdata=%.2f'%(event.x, event.xdata))
        if event.button and event.inaxes:
            self.move_t_to_x(self._t_grip, event.x, send_event=True)


class OverviewPanel(CanvasPanel):
    def __init__(self, parent, id=ID.CANVAS_PANEL_OVERVIEW):
        CanvasPanel.__init__(self, parent, id=id)
        
    # User interaction
#        self._dragging = False
        self._dt = 10
        self.canvas.mpl_connect('motion_notify_event', self.OnCanvasMotion)
        self.canvas.mpl_connect('button_press_event', self.OnCanvasMouseDown)
        self.canvas.mpl_connect('button_release_event', self.OnCanvasMouseUp)
    
    # Mouse Events
    def OnCanvasMotion(self, event):
        if event.inaxes and hasattr(self, '_t0'):
            self.set_marker_t1(event.xdata)            
    def OnCanvasMouseDown(self, event):
#        logging.debug('mouse down t=%.2f'%event.xdata)
        if event.inaxes:
            self._t0 = event.xdata
#            self._dragging = True
    def OnCanvasMouseUp(self, event):
#        logging.debug('mouse up t=%.2f'%event.xdata)
        if hasattr(self, '_t0'):
            logging.debug('has t0')
            if event.inaxes:
                if self._t0 == event.xdata:
                    self.set_marker_pos(event.xdata)
                else:
                    self.set_marker_t1(event.xdata)
            del self._t0
    
    def OnPaint(self, event):
        logging.debug('%s: OnPaint!!!' % self.__class__.__name__)
        if self.marker:
            self.marker.remove()
#            self.figure.remove(self.marker)
#            self.ax.remove(self.marker)
        self.canvas.draw()
        self._background = self.canvas.copy_from_bbox(self.ax.bbox)
        self._ylim = self.ax.get_ylim()
        if self.marker:
            self.ax.add_artist(self.marker)
            self.draw_marker()
        event.Skip()
    
    def plot_overview(self, visualizers):
        self.figure.clf()
        
        ax = self.ax = self.figure.add_axes([0, .1, 1, .9], frame_on=False)
        for v in visualizers:
            v.toax(ax, None, None)
        
        self.marker = None
        self._tmin = t0 = v.tstart
        self._tmax = t1 = v.tend
        ax.set_xlim(t0, t1)
    
    def set_marker_pos(self, t):
        dt = self._dt / 2
        t1 = t - dt
        t2 = t + dt
        if t1 < self._tmin:
            t2 = min(t2 + (self._tmin - t1), self._tmax)
            t1 = self._tmin
        elif t2 > self._tmax:
            t1 = max(t1 - (t2 - self._tmax), self._tmin)
            t2 = self._tmax
        self.zoom(t1, t2)
    
    def set_marker_t1(self, t1):
        "assumes that self._t0 is set"
        self.zoom(self._t0, t1)
    
    def zoom(self, t1, t2, send_event=True):
        if t1 == t2:
            self.set_marker_pos(t1)
            return
        t_start = min(t1, t2)
        t_end = max(t1, t2)
        
        self.canvas.restore_region(self._background)
        
        if self.marker:
#            logging.debug('marker:  just updating')
            xy = self.marker.get_xy()
            xy[0,0] = t1
            xy[1,0] = t1
            xy[2,0] = t2
            xy[3,0] = t2
            xy[4,0] = t1
#            logging.debug(xy)
        else:
            self.marker = self.ax.axvspan(t_start, t_end, edgecolor='r', 
                                          hatch='/', fill=False, aa=False,
                                          zorder=0) #, alpha=.1)# fill=False)
        self._dt = t_end - t_start
        self.ax.set_xlim(self._tmin, self._tmax)
        self.ax.set_ylim(*self._ylim)
        self.draw_marker()
        
        if send_event:
            self.SendFigurePosEvent(t1, t2)
#        ax.set_xlim(*self.viewer.t_lim)
    def draw_marker(self):
        try:
            self.ax.draw_artist(self.marker)
        except Exception:
            logging.debug('DRAW exception: %s'%Exception)
        else:
            pass
        self.canvas.blit(self.ax.bbox)
    def del_marker(self):
        if self.marker:
            self.figure.remove(self.marker)
            self.marker = None
            self.canvas.draw()
        
