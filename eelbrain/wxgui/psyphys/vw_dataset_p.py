"""
not used at the moment

"""

import threading, logging #, multiprocessing

import numpy as np

from eelbrain.signal_processing import param

import wx, wxmpl, wx.combo
import ID
from wx_base import Icon



class _plotter_thread(threading.Thread):
    def __init__(self, fig_panel, name="Plotter", plot_preview=False):
        threading.Thread.__init__(self, name=name)
        self.fig_panel = fig_panel
        self.plot_needs_update = threading.Event()
        self._suicidal = False
        self.plot_preview = plot_preview
        self.name = self.__class__.__name__
        self.start()
    def run(self):
        while True:
            logging.debug("plotter {cn} about to wait()".format(cn=self.name))
            self.plot_needs_update.wait()
            self.plot_needs_update.clear()
            logging.debug("start {cn} update".format(cn=self.name))
            if self._suicidal:
                break
            else:
                self.plot()
        logging.debug("plotter {cn} terminating".format(cn=self.name))
    def kill(self):
        self._suicidal = True
        self.plot_needs_update.set()
    def update(self):
        self.plot_needs_update.set()
    def block_update(self):
        self.plot_needs_update.clear()
    def draw(self):
        try:
            self.fig_panel.draw()
            msg = " -> successful plot {cn} draw()"
        except Exception, exc:
            msg = "PLOTTER {cn} draw() exception: {e1}, {e2}"
            msg = msg.format(cn='{cn}', e1=Exception, e2=exc)
        logging.debug(msg.format(cn=self.name))
    def plot(self):
        "SUBCLASS!"
        pass
    def set_preview_plotting(self, plotting):
        self.plot_preview = plotting
        

# MARK: Main Window

class _ListCtrlComboPopup(wx.ListCtrl, wx.combo.ComboPopup):
    def __init__(self, combo_ctrl):
        self.combo_ctrl = combo_ctrl
        # Since we are using multiple inheritance, and don't know yet
        # which window is to be the parent, we'll do 2-phase create of
        # the ListCtrl instead, and call its Create method later in
        # our Create method.  (See Create below.)
        self.PostCreate(wx.PreListCtrl())

        # Also init the ComboPopup base class.
        wx.combo.ComboPopup.__init__(self)


    def AddItem(self, txt):
        self.InsertStringItem(self.GetItemCount(), txt)

    def OnMotion(self, evt):
        item, flags = self.HitTest(evt.GetPosition())
        if item >= 0:
            self.Select(item)
            self.curitem = item

    def OnLeftDown(self, evt):
        self.value = self.curitem
        self.Dismiss()


    # The following methods are those that are overridable from the
    # ComboPopup base class.  Most of them are not required, but all
    # are shown here for demonstration purposes.


    # This is called immediately after construction finishes.  You can
    # use self.GetCombo if needed to get to the ComboCtrl instance.
    def Init(self):
        self.value = 0
        self.curitem = 0
        self.current_selection = 0

    # Create the popup child control.  Return true for success.
    def Create(self, parent):
        wx.ListCtrl.Create(self, parent,
                           style=wx.LC_LIST|wx.LC_SINGLE_SEL|wx.SIMPLE_BORDER)
        self.Bind(wx.EVT_MOTION, self.OnMotion)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.Bind(wx.EVT_BUTTON, self.OnSet)
        self.Bind(wx.EVT_LEFT_UP, self.OnSet)
        return True
    def OnSet(self, event):
        logging.debug("SET")
    # Return the widget that is to be used for the popup
    def GetControl(self):
        return self

    # Called just prior to displaying the popup, you can use it to
    # 'select' the current item.
    def SetStringValue(self, val):
        idx = self.FindItem(-1, val)
        if idx != wx.NOT_FOUND:
            self.Select(idx)

    # Return a string representation of the current item.
    def GetStringValue(self):
        logging.debug("GetStringValue")
        if self.value >= 0:
            if self.value != self.current_selection:
                self.combo_ctrl.SelectionChanged(self.value)
                self.current_selection = self.value
            return self.GetItemText(self.value)
        return ""

    # Called immediately after the popup is shown
    def OnPopup(self):
        wx.combo.ComboPopup.OnPopup(self)

    # Called when popup is dismissed
    def OnDismiss(self):
        wx.combo.ComboPopup.OnDismiss(self)

    # This is called to custom paint in the combo control itself
    # (ie. not the popup).  Default implementation draws value as
    # string.
    def PaintComboControl(self, dc, rect):
        wx.combo.ComboPopup.PaintComboControl(self, dc, rect)

    # Receives key events from the parent ComboCtrl.  Events not
    # handled should be skipped, as usual.
    def OnComboKeyEvent(self, event):
        logging.debug("KEY EVENT")
        wx.combo.ComboPopup.OnComboKeyEvent(self, event)

    def OnComboDoubleClick(self):
        wx.combo.ComboPopup.OnComboDoubleClick(self)

    # Return final size of popup. Called on every popup, just prior to OnPopup.
    # minWidth = preferred minimum width for window
    # prefHeight = preferred height. Only applies if > 0,
    # maxHeight = max height for window, as limited by screen size
    #   and should only be rounded down, if necessary.
    def GetAdjustedSize(self, minWidth, prefHeight, maxHeight):
        return wx.combo.ComboPopup.GetAdjustedSize(self, minWidth, prefHeight, maxHeight)
   
class _ComboSegmentSelector(wx.combo.ComboCtrl):
    def __init__(self, segments, parent, viewer, style=wx.CB_READONLY|wx.LEFT, 
                 size=(250,-1)):
        self.viewer = viewer
        wx.combo.ComboCtrl.__init__(self, parent, style=style, size=size)
        self.seg_list = _ListCtrlComboPopup(self)
        self.SetPopupControl(self.seg_list)
        for seg in segments:
            self.seg_list.AddItem(seg.name)
        #self.Bind(wx.EVT_CHOICE, self.OnSet)
        #self.Bind(wx.EVT_LEFT_UP, self.OnSet)
        #self.Bind(wx.EVT_COMBOBOX, self.OnSet)
    def SelectionChanged(self, i):
        logging.debug(" COMBO SLEECRION CHANGED to %s"%i)
        self.viewer.OnExampleSet(i)
    def OnButtonClick(self):
        logging.debug("OnButtonClick")
        wx.combo.ComboCtrl.OnButtonClick(self)
    def SetTo(self, i):
        logging.debug("!! setto" + str(dir(self)))
        self.SetText(self.seg_list.GetItemText(i))
        self.seg_list.Select(i)
        
        
class physio_dataset_viewer(wx.Frame):
    def __init__(self, dataset, parent=None, id=-1):
        title = self.title_stem = dataset.name
        wx.Frame.__init__(self, parent, id, title, size=(800,600))
        self.dataset = dataset
        # Toolbar
        tb = self.CreateToolBar(wx.TB_HORIZONTAL)
        tb.SetToolBitmapSize(size=(32,32))
        
            # select file
        tb.AddLabelTool(wx.ID_BACKWARD, "Back", Icon("tango/actions/go-previous"))
        self.Bind(wx.EVT_TOOL, self.OnExampleBack, id=wx.ID_BACKWARD)
        tb.AddLabelTool(wx.ID_FORWARD, "Next", Icon("tango/actions/go-next"))
        self.Bind(wx.EVT_TOOL, self.OnExampleNext, id=wx.ID_FORWARD)
        
        css = self.css = _ComboSegmentSelector(dataset.segments, tb, self)
        css.Bind(wx.EVT_TEXT, self.OnComboSegSel)
        tb.AddControl(css)
        
            # Display
        tb.AddSeparator()
        checkbox = wx.CheckBox(tb, ID.SHOW_SOURCE, "Show Source")
        tb.AddControl(checkbox)
        self.Bind(wx.EVT_CHECKBOX, self.OnCheckBox)
        
        tb.AddLabelTool(wx.ID_ZOOM_IN, "Zoom In", Icon("actions/zoom_in"))
        tb.AddLabelTool(wx.ID_ZOOM_OUT, "Zoom Out", Icon("actions/zoom_out"))
        self.Bind(wx.EVT_TOOL, self.OnZoom, id=wx.ID_ZOOM_IN)
        self.Bind(wx.EVT_TOOL, self.OnZoom, id=wx.ID_ZOOM_OUT)
        
        tb.AddLabelTool(wx.ID_UP, "Move Left", Icon("tango/actions/media-seek-backward"))
        tb.AddLabelTool(wx.ID_DOWN, "Move Left", Icon("tango/actions/media-seek-forward"))
        self.Bind(wx.EVT_TOOL, self.OnMove, id=wx.ID_UP)
        self.Bind(wx.EVT_TOOL, self.OnMove, id=wx.ID_DOWN)
        
        tb.AddLabelTool(wx.ID_REFRESH, "Refresh", Icon("tango/actions/view-refresh"))
        self.Bind(wx.EVT_TOOL, self.DataChanged, id=wx.ID_REFRESH)
        
        tb.Realize()
        #
        s1 = self.splitter_1 = wx.SplitterWindow(self, -1)#, style=wx.SP_3DSASH) 
        s2 = self.splitter_2 = wx.SplitterWindow(s1, -1)
        p1 = self.detailview = _physio_detailview_panel(s1)
        p2a = self.overview = _physio_overview_panel(s2, dataset, p1) 
        p2b = self.settings = _dataset_settings_panel(dataset.p, s2, 
                                                      p2a.plotter.data_update)
        logging.debug("VW: pabels created")
        s1.SplitHorizontally(p1, s2, 350)
        s2.SplitHorizontally(p2a, p2b, -50)
        s1.SetSashGravity(.8)
        s2.SetSashGravity(1.)
        #
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        self.Show()
        #
        self.example_i = 0
        self.DataChanged()
    def OnZoom(self, event):
        id = event.GetId()
        if id == wx.ID_ZOOM_IN:
            self.overview.plotter.zoom(1.33)
        elif id == wx.ID_ZOOM_OUT:
            logging.debug("zoom out")
            self.overview.plotter.zoom(.66)
        else:
            raise("ID NOT GOOD!!")
    def OnMove(self, event=None):
        if event.GetId() == wx.ID_UP: # left
            self.overview.plotter.move(-.25)
        else:
            self.overview.plotter.move(.25)
    def OnComboSegSel(self, event):
        logging.debug("COMBO")
        logging.debug(dir(event))
    def OnClose(self, event):
        logging.debug("Window Closing")
        self.overview.plotter.kill()
        self.detailview.plotter.kill()
        event.Skip()
    def OnExampleNext(self, event):
        if self.example_i < len(self.dataset.segments)-1:
            self.example_i += 1
        else:
            self.example_i = 0
        self.DataChanged()
    def OnExampleBack(self, event):
        if self.example_i > 0:
            self.example_i -= 1
        else:
            self.example_i = len(self.dataset.segments)-1
        self.DataChanged()
    def OnExampleSet(self, i):
        if 0 <= i < len(self.dataset.segments):
            self.example_i = i
            self.DataChanged()
        else:
            logging.error(" Viewer OnExampleSet Index out of range")
    def DataChanged(self, event=None):
        logging.debug("VW: DataChanged()")
        self.css.SetTo(self.example_i)
        seg = self.dataset.segments[self.example_i]
        self.SetTitle("{0}: {1}".format(self.title_stem, seg.name))
        self.overview.set_segment(seg)
    def OnCheckBox(self, event):
        if event.GetId() == ID.SHOW_SOURCE:
            plotting = event.Checked()
            self.overview.plotter.set_preview_plotting(plotting)
            self.detailview.plotter.set_preview_plotting(plotting)


# MARK: overview view

class _physio_overview_panel(wxmpl.PlotPanel):
    def __init__(self, parent, dataset, detail_view, id=-1):
        wxmpl.PlotPanel.__init__(self, parent, id, zoom=False, selection=True,
                                 crosshairs=False, )
        fig = self.fig = self.get_figure()
        ax = self.ax = fig.add_axes([0, .1, 1, .9])
        # set up rendering pipeline
        self.plotter = _plotter_overview(self, dataset, detail_view.plotter)
        wxmpl.EVT_POINT(self, -1, self.OnTstart)
        wxmpl.EVT_SELECTION(self, -1, self.OnTinterval)
        #cid = fig.canvas.mpl_connect('button_press_event', self.OnClick)
    def set_segment(self, segment):
        self.segment = segment
        self.plotter.update()
    #def update_detailview(self):
    #    detail_data = self.plotter.data[start, end]
    def OnClick(self, event):
        print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
                    event.button, event.x, event.y, event.xdata, event.ydata)
        print dir(event)
    def OnTstart(self, event):
        #print "notify_point", x, y
        self.plotter.set_start(event.xdata)
    def OnTinterval(self, event):
        #print "notify_selection", x1, y1, x2, y2
        self.plotter.set_interval(event.x1data, event.x2data)


class _plotter_overview(_plotter_thread):
    def __init__(self, fig_panel, dataset, detail_view_plotter):
        self.detail_view_plotter = detail_view_plotter
        self.dataset = dataset
        self.seg = None
        self.t_min = False # whether t is represented in min (or sec)
        self.detail_start = 0
        self.detail_end = 30
        self.data_needs_update = False # flag indicating whether thread should update the data
        _plotter_thread.__init__(self, fig_panel, name="Plotter_overview")
    def set_start(self, t):
        detail_interval = self.detail_end - self.detail_start
        self.detail_start = t
        self.detail_end = t + detail_interval
        self.update()
    def set_interval(self, t1, t2):
        self.detail_start = t1
        self.detail_end = t2
        self.update()
    def zoom(self, factor):
        t1 = self.detail_start
        t2 = self.detail_end
        diff = ((t2 - t1) * (factor-1)) / 2.
        self.detail_start = t1 + diff
        self.detail_end = t2 - diff
        self.update()
    def move(self, factor):
        t1 = self.detail_start
        t2 = self.detail_end
        diff = (t2 - t1) * factor
        self.detail_start = t1 + diff
        self.detail_end = t2 + diff
        self.update()
    def set_preview_plotting(self, plotting):
        self.plot_preview = plotting
        self.data_update()
    def data_update(self):
        self.data_needs_update = True
        self.update()
    def plot(self):
        # acquire data and properties
        seg = self.fig_panel.segment
        samplingrate = seg['samplingrate']
        if (seg != self.seg) or self.data_needs_update:
            self.data_needs_update = False
            self.seg = seg
            t_start = seg.t_start
            t_end = seg.t_end
            T = np.arange(t_start, t_end, 1./samplingrate)
            xlabel = 'T [Seconds]'
            self.t_to_i = samplingrate
            self.t_min = False
            self.xlen = len(T)
            ############
            # adjust detail t
            diff = self.detail_end - T[-1]
            if diff > 0:
                self.detail_end = T[-1]
                self.detail_start = max(T[0], self.detail_start - diff)
            elif self.detail_start < T[0]:
                diff2 = T[0] - self.detail_start
                self.detail_end = min(self.detail_end + diff2, T[-1])
                self.detail_start = T[0]                
            t1 = self.detail_start
            t2 = self.detail_end            
            #get data
            if seg['data_type'] == 'event':
                data = seg.asdata()
            else:
                if self.plot_preview:
                    data = self.dataset._process_segment(seg, preview=True)
                else:
                    data = seg.data
            if self.plot_preview:
                # scaling
                data_min = data.min(0)
                data_min[0] = 0
                data -= data_min
                data_max = data.max(0)
                data_max /= data_max[0]
                data_max[0] = 1
                data /= data_max
            self.data = data
            # plot
            ax = self.fig_panel.ax
            ax.cla()
            d_min, d_max = min(data[:,0]), max(data[:,0])
            for i,c in zip(range(data.shape[1]), ['b','.5','y','g']):
                if np.max(data[:,i]) == np.nan:
                    ax.text(T[0], 0, "Plotting Error (nan)", color='r',
                            horizontalalignment='left', verticalalignment='bottom')
                else:
                    print 'MAX', np.max(data[:,i])
                    ax.plot(T, data[:,i], c, zorder=10-i)
            ax.set_xlabel(xlabel)
            # marker
            m = ax.axvspan(t1, t2, edgecolor='r', color='r', zorder=0)#, alpha=.1)# fill=False)
            self.detail_marker = m
            ax.set_xlim(T[0], T[-1])
        else:
        # modify marker
            #print dir(self.detail_marker)
            #print self.detail_marker.get_xy()
            xy = self.detail_marker.get_xy()
            t1, t2 = self.detail_start, self.detail_end
            print xy
            xy[:,0] = [t1,t1,t2,t2,t1]
            #self.detail_marker.set_xy
        # modify detail view
        i1, i2 = t1*self.t_to_i, t2*self.t_to_i
        self.detail_view_plotter.set_data(self.data[i1:i2], t1, t2, 
                                          int(t1*samplingrate), self.xlen)
        # draw
        self.draw()



# MARK: detail view

class _physio_detailview_panel(wx.Panel):#wx.ScrolledWindow):#, wxmpl.PlotPanel):
    def __init__(self, parent, id=-1):
        #wxmpl.PlotPanel.__init__(self, parent, -1, zoom=False, selection=False,
        #                                                    crosshairs=False)
        #wx.ScrolledWindow.__init__(self, parent, id)
        wx.Panel.__init__(self, parent, id)
        #fig = self.fig = self.get_figure()
        # # #
        pp = self.pp = wxmpl.PlotPanel(self, -1, zoom=False, selection=False,
                                           crosshairs=False)
        pp.SetScrollbars = self.SetScrollbars
        #self.SetTargetWindow(pp)
        fig = pp.fig = pp.get_figure()
        ax = pp.ax = fig.add_axes([0.03, .05, .97, .95])
        self.plotter = _plotter_data(pp)
        # # #
        #self.SetScrollPageSize(wx.HORIZONTAL, 1000)
        # sizer
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.pp, 1, wx.EXPAND)
        self.SetSizerAndFit(sizer)
        
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SCROLLBAR, self.OnScroll)
        self.Bind(wx.EVT_SCROLLWIN_BOTTOM, self.OnScroll)
        self.Bind(wx.EVT_SCROLLWIN_THUMBRELEASE, self.OnScroll)
        self.Bind(wx.EVT_MOUSEWHEEL, self.OnScroll)
    def OnPaint(self, event):
        print "PAINT"
    def OnScroll(self, event):
        print "SCROLL"
    def SetScrollbars(self, *args, **kwargs):
        pass # repeated call caused bus error

class _plotter_data(_plotter_thread):
    def set_data(self, data, t1, t2, x1, xlen):
        self.block_update()
        self.data = data
        self.t1 = t1
        self.t2 = t2
        self.fig_panel.SetScrollbars(1, 0, xlen, 0, x1, 0)
        self.update()
    def plot(self):
        # plot
        ax = self.fig_panel.ax
        ax.cla()
        T = np.linspace(self.t1, self.t2, len(self.data))
        data = self.data
        for i,c in zip(range(data.shape[1]), ['b','.5','y','g']):
            ax.plot(T, data[:,i], c, zorder=10-i)
        ax.set_xlim(self.t1, self.t2)
        # draw
        self.draw()
        

# MARK: settings panel

class _dataset_settings_panel(wx.Panel):
    def __init__(self, params, parent, plot_update_func, id=-1):
        wx.Panel.__init__(self, parent, id)
        self.plot_update = plot_update_func
        # param controls
        gbs = self.gbs = wx.GridBagSizer(2, len(params))
        self.params = {}
        for i, (name, param) in enumerate(params._params.iteritems()):
            gbs.Add(wx.StaticText(self, -1, name), (0, i), flag=wx.ALIGN_RIGHT)
            if type(param) == param.ChoiceParam:
                ctrl = wx.Choice(self, i, choices=param.choices)
                ctrl.SetSelection(param.n())
            else:
                ctrl = wx.TextCtrl(self, i, unicode(param.get()), 
                                   style=wx.TE_PROCESS_ENTER) #|wx.TE_PROCESS_TAB)
            gbs.Add(ctrl, (1,i), flag=wx.EXPAND)
            self.params[i] = param
        self.SetSizerAndFit(gbs)
        # Bindings
        self.Bind(wx.EVT_CHOICE, self.OnParamEnter)
        self.Bind(wx.EVT_CHAR, self.OnChar)
        self.Bind(wx.EVT_TEXT_ENTER, self.OnParamEnter)
        #self.Bind(wx.EVT_TEXT, self.OnParamEnter)
    def OnChar(self, event):
        logging.debug("ONCHAR: %s"%event.GetString())
        if event.GetKeyCode() == 9:
            self.OnParamEnter(event)
        event.Skip()
    def OnParamEnter(self, event):
        logging.debug("Param changed")
        Id = event.GetId() # ID
        arg = event.GetString()
        param = self.params[Id]
        #try:
        param.set(arg)
        #except Exception, exc:
        #    print "DSVIEWER: PARAM {0} ERROR: {1}".format(param.name, arg)
        self.plot_update()
        #print dir(event)
        ## call _compile ()