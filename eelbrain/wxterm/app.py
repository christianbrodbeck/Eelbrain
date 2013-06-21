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
    def __init__(self, redirect=False, filename=None, **shell_kwargs):
        """
        redirect : bool
            Redirect sys.stdout and sys.stderr; Redirects the output of
            internal ``print`` commands

        filename :
            Target for redirected

        """
        self.shell_kwargs = shell_kwargs
        wx.App.__init__(self, redirect=redirect, filename=filename)

    def OnInit(self):
        self.shell = ShellFrame(None, title='Eelbrain Shell',
                                **self.shell_kwargs)
        self.SetTopWindow(self.shell)
        self.shell.Show()
        if wx.__version__ >= '2.9':
            dlg = ModalDummyDialog('dsa', 'dsa')
            wx.CallLater(200, dlg.CloseAndDestroy)
            dlg.ShowModal()

        if len(sys.argv) > 1:
            for arg in sys.argv[1:]:
                self.shell.OnFileOpen(None, path=arg)

        return True

    def MacOpenFile(self, fname):
        if not fname.endswith('eelbrain_run.py'):
            self.MacOpenFiles([fname])

    def MacOpenFiles(self, filenames):
        logging.debug("MacOpenFiles: %s" % repr(filenames))
        for filename in filenames:
            if os.path.isfile(filename):
                self.shell.OnFileOpen(path=filename)
