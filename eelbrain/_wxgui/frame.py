import wx


class EelbrainWindow(object):
    # Frame subclass to support UI Update

    def OnUpdateUIBackward(self, event):
        event.Enable(False)

    def OnUpdateUIClear(self, event):
        event.Enable(False)

    def OnUpdateUIClose(self, event):
        event.Enable(True)

    def OnUpdateUIForward(self, event):
        event.Enable(False)

    def OnUpdateUIOpen(self, event):
        event.Enable(False)

    def OnUpdateUIPlotRange(self, event):
        event.Enable(False)
        event.Check(False)

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


class EelbrainFrame(wx.Frame, EelbrainWindow):

    pass


class EelbrainDialog(wx.Dialog, EelbrainWindow):

    pass
