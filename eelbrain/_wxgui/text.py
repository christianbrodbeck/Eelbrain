# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import wx

from .frame import EelbrainFrame


class TextFrame(EelbrainFrame):
    "Read-only text frame, shows itself"
    def __init__(self, parent, title, text, *args, **kwargs):
        super(TextFrame, self).__init__(parent, title=title, *args, **kwargs)
        self.text = wx.TextCtrl(self, wx.ID_ANY, text,
                                style=wx.TE_MULTILINE|wx.TE_READONLY)
        self.Show()
