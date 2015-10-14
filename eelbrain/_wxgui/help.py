# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Help Viewer"""
import wx
from wx.html2 import WebView

from ..fmtxt import _html_doc_template
from .frame import EelbrainFrame


def show_help_txt(text, parent, title=""):
    "Show help frame with text in mono-spaced font"
    lines = (line.replace(' ', '&nbsp;') for line in text.splitlines())
    body = "<code>%s</code>" % '<br>'.join(lines)
    html = _html_doc_template.format(title="Help: %s" % title, body=body)
    frame = HelpFrame(parent)
    frame.SetPage(html, title)
    frame.Show()


class HelpFrame(EelbrainFrame):
    #  http://stackoverflow.com/a/10866495/166700
    def __init__(self, parent, *args, **kwargs):
        display_w, display_h = wx.DisplaySize()
        x = 0
        y = 25
        w = min(650, display_w)
        h = min(1000, display_h - y)
        EelbrainFrame.__init__(self, parent, pos=(x, y), size=(w, h), *args, **kwargs)
        self.webview = WebView.New(self)

    def SetPage(self, html, url):
        self.webview.SetPage(html, url)
        self.SetTitle(self.webview.GetCurrentTitle())
