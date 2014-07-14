"""
the main application for Eelbrain's wx-terminal

"""

import logging
import os
import sys

import wx

from .shell import ShellFrame


class ModalDummyDialog(wx.Dialog):
    """Modal Dialog to show in wxPython 2.9.x on startup which makes invisible
    windows visible.
    """
    def __init__(self, message, title):
        wx.Dialog.__init__(self, None, -1, title, size=(200, 200))
        self.CenterOnScreen(wx.BOTH)

    def CloseAndDestroy(self):
        self.Close()
        self.Destroy()


class MainApp(wx.App):
    """
    The main application (:py:class:`wx.App` subclass). Creates the shell
    instance.

    """
    def __init__(self, py2app):
        self.py2app = py2app
        wx.App.__init__(self)  # , redirect=redirect, filename=filename)

    def OnInit(self):
        self.shell = ShellFrame(app=self)
        self.SetTopWindow(self.shell)
        self.shell.Show()
        if self.py2app and wx.__version__ >= '2.9':
            dlg = ModalDummyDialog('dsa', 'dsa')
            wx.CallLater(200, dlg.CloseAndDestroy)
            dlg.ShowModal()

        if len(sys.argv) > 1:
            for arg in sys.argv[1:]:
                self.shell.OnFileOpen(None, path=arg)

        return True

    def BringWindowToFront(self):
        win = self.shell.get_active_window()
        if win is None:
            self.shell.Show()
            self.shell.Raise()
        else:
            win.Raise()

    def MacOpenFile(self, fname):
        if not fname.endswith('bin/eelbrain'):
            self.MacOpenFiles([fname])

    def MacOpenFiles(self, filenames):
        logging.debug("MacOpenFiles: %s" % repr(filenames))
        for filename in filenames:
            if os.path.isfile(filename):
                self.shell.OnFileOpen(path=filename)

    def MacReopenApp(self):
        """Called when the doc icon is clicked"""
        self.BringWindowToFront()

    def OnActivate(self, event):
        # if this is an activate event, rather than something else, like iconize.
        if event.GetActive():
            self.BringWindowToFront()
        event.Skip()
