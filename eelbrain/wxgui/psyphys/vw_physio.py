"""
wx structure:

- physio_dataset_viewer
    - _physio_view_panel_overview
    - _physio_view_panel_detailview
    - _dataset_settings_panel


"""

import logging

import wx.combo

from ...utils.basic import toList
from ...wxutils import Icon
import ID

import view_panels



class PhysioViewerFrame(wx.Frame):
    def __init__(self, parent, visualizers, i=0, id= -1, size=(800, 600)):
        title = self.title_stem = "Physio Viewer"  # dataset.name
        wx.Frame.__init__(self, parent, id, title, size=size)

        self.visualizers = visualizers = toList(visualizers)
        datasets = self.datasets = [v.dataset for v in visualizers]
        self.active_dataset = datasets[0]
#        self.visualizers.reverse()


    # Toolbar
        tb = self.CreateToolBar(wx.TB_HORIZONTAL)
        tb.SetToolBitmapSize(size=(32, 32))

        # select file
        tb.AddLabelTool(wx.ID_BACKWARD, "Back", Icon("tango/actions/go-previous"))
        self.Bind(wx.EVT_TOOL, self.OnBack, id=wx.ID_BACKWARD)
        tb.AddLabelTool(wx.ID_FORWARD, "Next", Icon("tango/actions/go-next"))
        self.Bind(wx.EVT_TOOL, self.OnNext, id=wx.ID_FORWARD)

        css = self.css = _ComboSegmentSelector(tb, datasets[0].segments, self)
        css.Bind(wx.EVT_TEXT, self.OnComboSegSel)
        tb.AddControl(css)

        # Display
        tb.AddSeparator()
#        checkbox = wx.CheckBox(tb, wx_base.ID_SHOW_SOURCE, "Show Source")
#        tb.AddControl(checkbox)
#        self.Bind(wx.EVT_CHECKBOX, self.OnCheckBox)

        tb.AddLabelTool(wx.ID_ZOOM_IN, "Zoom In", Icon("actions/zoom-in"))
        tb.AddLabelTool(wx.ID_ZOOM_OUT, "Zoom Out", Icon("actions/zoom-out"))
        self.Bind(wx.EVT_TOOL, self.OnZoom, id=wx.ID_ZOOM_IN)
        self.Bind(wx.EVT_TOOL, self.OnZoom, id=wx.ID_ZOOM_OUT)

        tb.AddLabelTool(wx.ID_UP, "Move Left", Icon("tango/actions/media-seek-backward"))
        tb.AddLabelTool(wx.ID_DOWN, "Move Left", Icon("tango/actions/media-seek-forward"))
        self.Bind(wx.EVT_TOOL, self.OnMove, id=wx.ID_UP)
        self.Bind(wx.EVT_TOOL, self.OnMove, id=wx.ID_DOWN)

        tb.AddLabelTool(wx.ID_REFRESH, "Refresh", Icon("tango/actions/view-refresh"))
        self.Bind(wx.EVT_TOOL, self.OnReloadData, id=wx.ID_REFRESH)

        tb.Realize()


    # EVTs
        self.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
        self.Bind(wx.EVT_CLOSE, self.OnClose)

    # Create Panels
        s1 = self.splitter_1 = wx.SplitterWindow(self, -1)  # , style=wx.SP_3DSASH)
        s1.SetSashGravity(1)
#        s2 = self.splitter_2 = wx.SplitterWindow(s1, -1)

        p1 = self.ZoomPanel = view_panels.ZoomPanel(s1)
        p2 = self.OverviewPanel = view_panels.OverviewPanel(s1)

        self.Bind(view_panels.EVT_FIGURE_POS, p1.OnFigurePosEvent, id=ID.CANVAS_PANEL_OVERVIEW)  # source=p2)
        self.Bind(view_panels.EVT_FIGURE_POS, p2.OnFigurePosEvent, id=ID.CANVAS_PANEL_ZOOM)  # source=p1)
#        p1 = self.detailview = _physio_view_panel_detailview(s1, self)
#        p2a = self.overview  = _physio_view_panel_overview(s2, self)
#        p2b = self.settings  = wx.Panel(s2, id=wx.ID_ANY)
#        p2b = self.settings  = _dataset_settings_panel(datasets[0].p, s2, self)
        logging.debug("PHYSIO Panels created")

        s1.SplitHorizontally(p1, p2, 300)
        self.set_segment(0)
        self.Show()

#        s2.SplitHorizontally(p2a, p2b, -50)
#        s1.SetSashGravity(.8)
#        s2.SetSashGravity(1.)

    # GUI functions
    def OnKeyDown(self, event):
        key = event.GetKeyCode()
        logging.debug(" PhysioViewer OnKeyDown: {0}".format(key))
        event.Skip()
#        if event.MetaDown():
#            logging.debug("event.KeyCode=%s"%event.KeyCode)
#        else:
#            event.Skip()
    def OnZoom(self, event):
        id = event.GetId()
        if id == wx.ID_ZOOM_IN:
            self.ZoomPanel.relative_zoom(1.33)
        elif id == wx.ID_ZOOM_OUT:
            logging.debug("zoom out")
            self.ZoomPanel.relative_zoom(.66)
        else:
            raise ValueError("ID NOT GOOD!!")
    def OnMove(self, event=None):
        if event.GetId() == wx.ID_UP:  # left
            self.ZoomPanel.relative_move(-.25)
        else:
            self.ZoomPanel.relative_move(.25)
    def OnComboSegSel(self, event):
        logging.debug("COMBO")
        logging.debug(dir(event))
    def OnClose(self, event):
        logging.debug(" --> Window Closing")
        event.Skip()
    def OnNext(self, event):
        self.set_segment(self.i + 1)
    def OnBack(self, event):
        self.set_segment(self.i - 1)
    def OnReloadData(self, event=None):
        logging.debug("VW: OnReloadData()")
        self.update_data()
#    def OnCheckBox(self, event):
#        if event.GetId() == wx_base.ID_SHOW_SOURCE:
#            plotting = event.Checked()
#            self.overview.plotter.set_preview_plotting(plotting)
#            self.detailview.plotter.set_preview_plotting(plotting)

    # manage data
    def set_segment(self, i):
        "call when changing segment"
        max_i = len(self.datasets[0]) - 1
        if i > max_i:
            i = 0
        elif i < 0:
            i = max_i
        self.i = i
        self.css.SetTo(self.i)
        self.update_data()
    def update_data(self):
        "call whenever data has to be reloaded"
        seg = self.datasets[0].segments[self.i]
        self.SetTitle("{0}: {1}".format(self.title_stem, seg.name))
        for v in self.visualizers:
            v.set_segment(self.i)

        tmin = seg.tstart
        tmax = seg.tend
#        if self.t_start < tmin:
#            self.set_start(tmin)
#        if self.t_end > tmax:
#            self.set_end(tmax)
        self.t_lim = (tmin, tmax)

        # plot the new data
        self.OverviewPanel.plot_overview(self.visualizers)
        self.OverviewPanel.Refresh()
        self.ZoomPanel.plot(self.visualizers)
        self.ZoomPanel.Refresh()



# Helpers

class _ComboSegmentSelector(wx.combo.ComboCtrl):
    def __init__(self, parent, segments, viewer, style=wx.CB_READONLY | wx.LEFT,
                 size=(250, -1)):
        wx.combo.ComboCtrl.__init__(self, parent, style=style, size=size)
        self.viewer = viewer
        self.popup_list = _ListCtrlComboPopup(self)
        self.SetPopupControl(self.popup_list)
        for seg in segments:
            self.popup_list.AddItem(seg.name)
        # self.Bind(wx.EVT_CHOICE, self.OnSet)
        # self.Bind(wx.EVT_LEFT_UP, self.OnSet)
        # self.Bind(wx.EVT_COMBOBOX, self.OnSet)
    def SelectionChanged(self, i):
#        logging.debug(" COMBO SLEECTION CHANGED to %s"%i)
        self.viewer.set_segment(i)
    def OnButtonClick(self):
#        logging.debug("OnButtonClick")
        wx.combo.ComboCtrl.OnButtonClick(self)
    def SetTo(self, i):
#        logging.debug("!! setto" + str(dir(self)))
        self.SetText(self.popup_list.GetItemText(i))
        self.popup_list.Select(i)


class _ListCtrlComboPopup(wx.ListCtrl, wx.combo.ComboPopup):
    """
    popup for selection of segment out of dataset

    """
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
                           style=wx.LC_LIST | wx.LC_SINGLE_SEL | wx.SIMPLE_BORDER)
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




class _dataset_settings_panel(wx.Panel):
    def __init__(self, params, parent, viewer, id= -1):
        wx.Panel.__init__(self, parent, id)
        self.viewer = viewer
        # param controls
        gbs = self.gbs = wx.GridBagSizer(2, len(params))
        self.params = {}
        for i, (name, par) in enumerate(params._params.iteritems()):
            gbs.Add(wx.StaticText(self, -1, name), (0, i), flag=wx.ALIGN_RIGHT)
            if par.__class__.__name__ == 'Choice':
                ctrl = wx.Choice(self, i, choices=par._options)
                ctrl.SetSelection(par.get_i())
            else:
#                if hasattr(param, '_get_repr'):
                txt = par.__value_repr__()
                ctrl = wx.TextCtrl(self, i, unicode(txt),
                                   style=wx.TE_PROCESS_ENTER)  # |wx.TE_PROCESS_TAB)
            gbs.Add(ctrl, (1, i), flag=wx.EXPAND)
            self.params[i] = par
        self.SetSizerAndFit(gbs)
        # Bindings
        self.Bind(wx.EVT_CHOICE, self.OnParamEnter)
#        self.Bind(wx.EVT_CHAR, self.OnChar)
        self.Bind(wx.EVT_TEXT_ENTER, self.OnParamEnter)
        # self.Bind(wx.EVT_TEXT, self.OnParamEnter)
    def OnChar(self, event):
        logging.debug("ONCHAR: %s" % event.GetString())
        if event.GetKeyCode() == 9:
            self.OnParamEnter(event)
        event.Skip()
    def OnParamEnter(self, event):
        logging.debug("Param changed")
        Id = event.GetId()  # ID
        arg = event.GetString()
        par = self.params[Id]
        try:
            arg = eval(arg)
        except:
            pass
        par(arg)
        # except Exception, exc:
        #    print "DSVIEWER: PARAM {0} ERROR: {1}".format(param.name, arg)
        ds = self.viewer.active_dataset
        if ds['data_type'] == 'event':
            ds.delete_segments()
            self.viewer.set_segment(self.viewer.i)
        else:
            self.viewer.update_data()
        # print dir(event)
