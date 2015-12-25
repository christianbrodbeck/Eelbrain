# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import wx
import wx.html

from .frame import EelbrainFrame


class TextFrame(EelbrainFrame):
    "Read-only text frame, shows itself"
    def __init__(self, parent, title, text, *args, **kwargs):
        super(TextFrame, self).__init__(parent, title=title, *args, **kwargs)
        self.text = wx.TextCtrl(self, wx.ID_ANY, text,
                                style=wx.TE_MULTILINE|wx.TE_READONLY)
        self.Show()


class HTMLWindow(wx.html.HtmlWindow):

    def OnLinkClicked(self, link):
        url = link.GetHref()
        cell = link.GetHtmlCell()
        self.Parent.OpenURL(url)


class HTMLFrame(EelbrainFrame):

    def __init__(self, parent, title, text, *args, **kwargs):
        EelbrainFrame.__init__(self, parent, title=title, *args, **kwargs)
        self.text = HTMLWindow(self, wx.ID_ANY, style=wx.VSCROLL)
        self.text.SetPage(text)
        self.Show()

    def OpenURL(self, url):
        raise NotImplementedError("url=%r" % url)
