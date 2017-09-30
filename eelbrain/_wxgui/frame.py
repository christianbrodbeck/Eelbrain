import sys

import wx

from .._wxutils import ID


IS_OSX = sys.platform == 'darwin'
IS_WINDOWS = sys.platform.startswith('win')
FOCUS_UI_UPDATE_FUNC_NAMES = {
    wx.ID_COPY: 'CanCopy',
    ID.COPY_AS_PNG: 'CanCopyPNG',
    wx.ID_CUT: 'CanCut',
    wx.ID_PASTE: 'CanPaste',
}


class EelbrainWindow(object):
    # Frame subclass to support UI Update

    def OnUpdateUIBackward(self, event):
        event.Enable(False)

    def OnUpdateUIClear(self, event):
        event.Enable(False)

    def OnUpdateUIClose(self, event):
        event.Enable(True)

    def OnUpdateUIDown(self, event):
        event.Enable(True)

    def OnUpdateUIForward(self, event):
        event.Enable(False)

    def OnUpdateUIFocus(self, event):
        func_name = FOCUS_UI_UPDATE_FUNC_NAMES[event.GetId()]
        win = self.FindFocus()
        func = getattr(win, func_name, None)
        if func is None:
            func = getattr(self, func_name, None)
            if func is None:
                event.Enable(False)
                return
        event.Enable(func())

    def OnUpdateUIOpen(self, event):
        event.Enable(False)

    def OnUpdateUIRedo(self, event):
        event.Enable(False)

    def OnUpdateUISave(self, event):
        event.Enable(False)

    def OnUpdateUISaveAs(self, event):
        event.Enable(False)

    def OnUpdateUISetLayout(self, event):
        event.Enable(False)

    def OnUpdateUISetVLim(self, event):
        event.Enable(False)

    def OnUpdateUIUndo(self, event):
        event.Enable(False)

    def OnUpdateUIUp(self, event):
        event.Enable(False)


class EelbrainFrame(wx.Frame, EelbrainWindow):

    def OnDrawCrosshairs(self, event):
        raise RuntimeError("%s can't draw crosshairs" % (self,))

    def OnUpdateUIDrawCrosshairs(self, event):
        event.Enable(False)
        event.Check(False)


class EelbrainDialog(wx.Dialog, EelbrainWindow):

    pass
