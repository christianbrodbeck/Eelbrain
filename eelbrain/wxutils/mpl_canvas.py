'''
Created on Dec 21, 2010

Mpl examples:
http://matplotlib.sourceforge.net/examples/user_interfaces/index.html


'''

import tempfile
import logging

import numpy as np
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.backends import backend_wx
from matplotlib.figure import Figure
import wx

from .._utils import ui
from . import ID
from .basic import Icon


class FigureCanvasPanel(FigureCanvasWxAgg):
    """
    Subclass mpl's Canvas to allow interaction with wx.py (such as copying the
    contents to the clipboard)
    start search at wx.py.frame.Frame.OnUpdateMenu()

    """
    _copy_as_pdf = True
    def __init__(self, parent, figsize=(8, 6), dpi=100, facecolor='w'):
        """
        figsize : (w, h)
            size in inches
        dpi : int
            dits per inch

        """
        self.figure = Figure(figsize, dpi, facecolor)
        FigureCanvasWxAgg.__init__(self, parent, -1, self.figure)
        self.Bind(wx.EVT_ENTER_WINDOW, self.ChangeCursor)
        self.Bind(wx.EVT_MENU, self.OnFileSave, id=wx.ID_SAVE)

    def CanCopy(self):
        "pretend to be wx.py.frame.Frame to enable 'copy' menu command"
        return True

    def bufferHasChanged(self):
        return True

    def ChangeCursor(self, event):
        "http://matplotlib.sourceforge.net/examples/user_interfaces/wxcursor_demo.html"
        self.SetCursor(wx.StockCursor(wx.CURSOR_CROSS))

#    def OnPaint(self, event):
#        self.draw()

    def Copy(self):
        if self._copy_as_pdf:
            try:
                if wx.TheClipboard.Open():
                    # same code in mpl_tools
                    path = tempfile.mktemp('.pdf')  # , text=True)
                    logging.debug("Temporary file created at: %s" % path)
                    self.figure.savefig(path)
                    # copy path
                    do = wx.FileDataObject()
                    do.AddFile(path)
                    wx.TheClipboard.SetData(do)
                    wx.TheClipboard.Close()
            except wx._core.PyAssertionError:
                wx.TheClipboard.Close()

        else:
            self.figure.set_facecolor((1, 1, 1))
            self.draw()
            self.Copy_to_Clipboard()

    def OnFileSaveAs(self, event):
        self.OnFileSave(event)

    def OnFileSave(self, event):
        """
        FIXME: maybe start search at wx.py.frame.Frame.OnUpdateMenu()

        """
        path = ui.ask_saveas("Save Figure", "Save the current figure. The "
                             "format is determined from the extension.", None)
        if path:
            self.figure.savefig(path)

    def redraw(self, axes=[], artists=[]):
        self.restore_region(self._background)
        for ax in axes:
            ax.draw_artist(ax)
            extent = ax.get_window_extent()
            self.blit(extent)
        for artist in artists:
            ax = artist.get_axes()
            # FIXME:
#            ax.draw_artist(artist)
#            extent = artist.get_window_extent(self.get_renderer()) # or self?
            # substitute redrawing whole ax
            ax.draw_artist(ax)
            extent = ax.get_window_extent()
            # end substitute
            self.blit(extent)

    def store_canvas(self):
        self._background = self.copy_from_bbox(self.figure.bbox)





class CanvasFrame(wx.Frame):
    """

    after:
    http://matplotlib.sourceforge.net/examples/user_interfaces/embedding_in_wx2.html

    """
    def __init__(self, parent=None, title="Matplotlib Frame",
                 figsize=(8, 6), dpi=50, facecolor='w', _EelFigure=None,
                 statusbar=True, toolbar=True, mpl_toolbar=False):
        size = (figsize[0] * dpi, figsize[1] * dpi + 45)
        wx.Frame.__init__(self, parent, -1, title=title, size=size)

    # set up the canvas
        # prepare the plot panel
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(sizer)
        self.canvas = FigureCanvasPanel(self, figsize, dpi, facecolor)
        sizer.Add(self.canvas, 1, flag=wx.EXPAND)
#        sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        sizer.Layout()

        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUI)
        self.Bind(wx.EVT_MENU, self.OnFileSave, id=wx.ID_SAVE)
        self.Bind(wx.EVT_MENU, self.OnFileSaveAs, id=wx.ID_SAVEAS)

        # get figure
        self.figure = self.canvas.figure

        if statusbar:
            self.CreateStatusBar()

        if toolbar:
            tb = self.CreateToolBar(wx.TB_HORIZONTAL)
            tb.SetToolBitmapSize(size=(32, 32))
            self.FillToolBar(tb, _EelFigure)
            tb.Realize()

        if mpl_toolbar:
            self.add_mpl_toolbar()

        self._eelfigure = _EelFigure
        self.Bind(wx.EVT_CLOSE, self.OnClose)

    def FillToolBar(self, tb, _EelFigure):
        "subclasses should call this after adding their own items"
        if hasattr(self.Parent, 'attach'):
            tb.AddLabelTool(ID.ATTACH, "Attach", Icon("actions/attach"))
            self.Bind(wx.EVT_TOOL, self.OnAttach, id=ID.ATTACH)

        tb.AddLabelTool(wx.ID_SAVE, "Save", Icon("tango/actions/document-save"),
                        shortHelp="Save Document")
        self.Bind(wx.EVT_TOOL, self.OnFileSave, id=wx.ID_SAVE)

        # intermediate, custom part
        if _EelFigure is not None:
            _EelFigure._fill_toolbar(tb)

        # right-most part
        if wx.__version__ >= '2.9':
            tb.AddStretchableSpace()
        else:
            tb.AddSeparator()

        tb.AddLabelTool(wx.ID_HELP, 'Help', Icon("tango/apps/help-browser"))
        self.Bind(wx.EVT_TOOL, self.OnHelp, id=wx.ID_HELP)

        tb.AddLabelTool(ID.FULLSCREEN, "Fullscreen", Icon("tango/actions/view-fullscreen"))
        logging.debug('filltb')
        self.Bind(wx.EVT_TOOL, self.OnShowFullScreen, id=ID.FULLSCREEN)

    def add_mpl_toolbar(self):
        self.toolbar = backend_wx.NavigationToolbar2Wx(self.canvas)
        self.toolbar.Realize()
        if 0:  # wx.Platform == '__WXMAC__':
            # Mac platform (OSX 10.3, MacPython) does not seem to cope with
            # having a toolbar in a sizer. This work-around gets the buttons
            # back, but at the expense of having the toolbar at the top
            self.SetToolBar(self.toolbar)
        else:
            # On Windows platform, default window size is incorrect, so set
            # toolbar width to figure width.
            tw, th = self.toolbar.GetSizeTuple()
            fw, fh = self.canvas.GetSizeTuple()
            # By adding toolbar in sizer, we are able to put it at the bottom
            # of the frame - so appearance is closer to GTK version.
            # As noted above, doesn't work for Mac.
            self.toolbar.SetSize(wx.Size(fw, th))
            self.Sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        # update the axes menu on the toolbar
        self.toolbar.update()

    def OnClose(self, event):
        # remove circular reference
        if hasattr(self, '_eelfigure') and self._eelfigure:
            del self._eelfigure._frame
            del self._eelfigure
        event.Skip()

    def OnAttach(self, event):
        items = {'p': self._eelfigure}
        self.Parent.attach(items, detach=False, _internal_call=True)
        self.Parent.Raise()

    def OnFileSave(self, event):
        self.OnFileSaveAs(event)

    def OnFileSaveAs(self, event):
        """
        FIXME: maybe start search at wx.py.frame.Frame.OnUpdateMenu()

        """
        path = ui.ask_saveas("Save Figure", "Save the current figure. The format is "
                             "determined from the extension.", None)
        if path:
            self.figure.savefig(path)

    def OnHelp(self, event):
        app = wx.GetApp()
        shell = getattr(app, 'shell', None)
        if hasattr(shell, 'help_lookup'):
            shell.help_lookup(self._eelfigure)
        else:
            print self.__doc__

    def OnShowFullScreen(self, event):
        self.ShowFullScreen(not self.IsFullScreen())

    def OnUpdateUI(self, event):
        Id = event.GetId()
#        logging.warning("CanvasFrame.OnUpdateUI")
        if Id == wx.ID_SAVE:
            event.Enable(True)
        elif Id == wx.ID_SAVEAS:
            event.Enable(True)
        else:
            pass

    def redraw(self, axes=[], artists=[]):
        self.canvas.redraw(axes=axes, artists=artists)

    def store_canvas(self):
        self.canvas.store_canvas()


class TestCanvas(CanvasFrame):
    "This is a minimal CanvasFrame subclass"
    def __init__(self, effect=10, mpl_toolbar=True):
        CanvasFrame.__init__(self, title="Test MPL Frame", mpl_toolbar=mpl_toolbar)
        self.plot()
        self.Show(effect)

    def plot(self):
        self.axes = self.figure.add_subplot(111)
        t = np.arange(0.0, 3.0, 0.01)
        s = np.sin(2 * np.pi * t)
        self.axes.plot(t, s)
