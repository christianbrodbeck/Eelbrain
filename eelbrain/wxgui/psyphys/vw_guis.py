'''
Created on Apr 10, 2011

@author: christian
'''

import wx, os
#import logging

from eelbrain.psyphys import visualizers as V

from vw_list import ListViewerFrame
from eelbrain.wxutils import Icon
#import ID




class CropGUI(ListViewerFrame):
    """
    List viewer adapted for cropping segments.
    
    Click on plots to mark t-start.
    Alt-click on plots to mark t-end.
    Right (alt-) click to remove cropping.
     
    """
    def __init__(self, parent, dataset, **kwargs):
        visualizers = [V.default(dataset.parent)]
        self.dataset = dataset
        self.span_kwargs = dict(ec='r', hatch='/', fill=False, aa=False, 
                                zorder=100, )
        ListViewerFrame.__init__(self, parent, visualizers, **kwargs)        
        
        # change cursor
        self._click_mode = 0 # 0=tstart;  1=tend
        self.canvas_panel.canvas.Bind(wx.EVT_MOTION, self.ChangeCursor)
        self.canvas_panel.canvas.Bind(wx.EVT_KEY_DOWN, self.ChangeCursor)
        self.canvas_panel.canvas.Bind(wx.EVT_KEY_UP, self.ChangeCursor)
        
        # capture clicks
        self.canvas_panel.canvas.mpl_connect('button_press_event', self.OnCanvasClick)
        
        
    # ToolBar
        tb = self.ToolBar
#        tb.AddSeparator()
#        self._tb_tstart = tb.AddCheckLabelTool(ID.SET_TSTART, "Set t-start", 
#                                               Icon("tango/actions/go-first"))
#        self.Bind(wx.EVT_TOOL, self.OnSelectTool, id=ID.SET_TSTART)
#        self._tb_tend = tb.AddCheckLabelTool(ID.SET_TEND, "Set t-end", 
#                                             Icon("tango/actions/go-last"))
#        self.Bind(wx.EVT_TOOL, self.OnSelectTool, id=ID.SET_TEND)
        if wx.__version__ >= '2.9':
            tb.AddStretchableSpace()
        else:
            tb.AddSeparator()
        
        tb.AddLabelTool(wx.ID_HELP, "Help", Icon("tango/apps/help-browser"))
        self.Bind(wx.EVT_TOOL, self.OnHelp, id=wx.ID_HELP)
        
        tb.Realize()
    def ChangeCursor(self, event):
        if event.AltDown():
            self._click_mode = 1
            self.canvas_panel.canvas.SetCursor(wx.StockCursor(wx.CURSOR_POINT_RIGHT))            
        else:
            self._click_mode = 0
            self.canvas_panel.canvas.SetCursor(wx.StockCursor(wx.CURSOR_POINT_LEFT))
    def OnHelp(self, event):
        lines = self.__doc__.strip().splitlines()
        msg = os.linesep.join(line.strip() for line in lines)
        dlg = wx.MessageDialog(self, msg, "CropGUI Help", wx.OK|wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destry() 
    def OnCanvasClick(self, event):
#        logging.debug('ButtonPress')
        ax = event.inaxes
        if ax:
#            logging.debug('Button %s' % event.button)
            if event.button == 1:
                value = t = event.xdata
            else:
                t = ax.get_xlim()[self._click_mode]
                value = None
            id = ax.segment_id
            
            span = self.markers[id][self._click_mode]
            xy = span.get_xy()
#            logging.debug(span.get_xy())
            xy[(2,3), 0] = t
#            logging.debug("t = %s" % t)
            span.set_xy(xy)
#            logging.debug(span.get_xy())
            
            if self._click_mode == 0:
                self.dataset.p.tstart[id] = value
            else:
                self.dataset.p.tend[id] = value
            self.canvas_panel.draw()
                
    def show_page(self, page):
        ListViewerFrame.show_page(self, page)
        
        # add markers:
        self.markers = {}
        for ax in self.axes:
            id = ax.segment_id
            segment = self.dataset.parent.segments_by_id[id]
            t0 = - segment['t0']
            tstart = self.dataset.p.tstart[id]
            tend = self.dataset.p.tend[id]
            t1 = segment.duration
            
            if tstart is None:
                tstart = t0
            if tstart == t0:
                tstart += .001 # axvspan not correctly drawn with x0==x1
            if tend is None:
                tend = t1
            if tend == t1:
                tend -= .001
            
            s1 = ax.axvspan(t0, tstart, **self.span_kwargs)
            s2 = ax.axvspan(t1, tend, **self.span_kwargs)
            self.markers[id] = s1, s2
        
        self.canvas_panel.draw()
            
            
#    def OnSelectTool(self, event):
#        logging.debug(event.GetEventType())
#        logging.debug(event.GetEventObject())
#        id = event.GetId()
#        logging.debug(id)
#        if id == ID.SET_TSTART:
#            if self._tb_tend.IsToggled():
#                self._tb_tend.Toggle()
#                self.ToolBar.Realize()
#        elif id == ID.SET_TEND:
#            if self._tb_tstart.IsToggled():
#                self._tb_tstart.Toggle()
#                self.ToolBar.Realize()
#        else:
#            raise ValueError('unrecognized command ID')
        