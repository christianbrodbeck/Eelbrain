import wx


class EelbrainFrame(wx.Frame):
    # Frame subclass to support UI Update

    def OnUpdateUIClear(self, event):
        event.Enable(False)

    def OnUpdateUIClose(self, event):
        event.Enable(True)

    def OnUpdateUICopy(self, event):
        event.Enable(False)

    def OnUpdateUISave(self, event):
        event.Enable(False)

    def OnUpdateUISaveAs(self, event):
        event.Enable(False)

    def OnUpdateUIRedo(self, event):
        event.Enable(False)

    def OnUpdateUIUndo(self, event):
        event.Enable(False)
