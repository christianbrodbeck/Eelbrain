'''
About dialog for Eelbrain wx-terminal


Created on Feb 20, 2012

@author: christian
'''

import wx
from wx.lib.agw import hyperlink

from .. import __version__
from .._wxutils import Icon


class AboutFrame(wx.MiniFrame):
    def __init__(self, parent):
        wx.MiniFrame.__init__(self, parent, -1, pos=(100, 100),
                              style=wx.CLOSE_BOX)

        sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(sizer)

    # LOGO / name / version
        img = Icon('eelbrain160')
        bmp = wx.StaticBitmap(self, -1, img)
        sizer.Add(bmp, 0, wx.LEFT | wx.TOP | wx.RIGHT | wx.ALIGN_CENTER_HORIZONTAL, 10)

        txt = "Eelbrain"
        text = wx.StaticText(self, -1, txt)
        font = text.GetFont()
        font.SetPointSize(30)
        text.SetFont(font)
        sizer.Add(text, 0, wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_HORIZONTAL, 10)

        txt = "Version %s" % __version__
        text = wx.StaticText(self, -1, txt)
        text.SetForegroundColour(wx.Colour(120, 120, 120))
        sizer.Add(text, 0, wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.ALIGN_CENTER_HORIZONTAL, 10)


    # Links
        link = hyperlink.HyperLinkCtrl(self, wx.ID_ANY, "Eelbrain Documentation",
                                       URL="https://pythonhosted.org/eelbrain/")
        sizer.Add(link, 0, wx.LEFT | wx.TOP | wx.RIGHT | wx.ALIGN_CENTER_HORIZONTAL, 10)

        link = hyperlink.HyperLinkCtrl(self, wx.ID_ANY, "Eelbrain on GitHub",
                                       URL="https://github.com/christianbrodbeck/Eelbrain/")
        sizer.Add(link, 0, wx.LEFT | wx.TOP | wx.RIGHT | wx.ALIGN_CENTER_HORIZONTAL, 10)


    # Text
        txt = (u"Eelbrain is \xa9 Christian M Brodbeck\n"
               u"all open-source components are\n\xa9 their owners")
        text = wx.StaticText(self, -1, txt, style=wx.ALIGN_CENTER)
#        txt = (u"Eelbrain is \xa9 Christian Brodbeck; All open-source "
#               u"components are \xa9 their owners.")
#        text = wx.TextCtrl(self, -1, txt,
#                           style = wx.TE_READONLY|wx.TE_MULTILINE)
#        text.SetMinSize(wx.Size(250, 50))
        self.text = text
        sizer.Add(text, 0, wx.ALL | wx.EXPAND, 10)


        # buttons
        h_sizer = wx.BoxSizer(wx.HORIZONTAL)

        button = wx.Button(self, -1, "About PyShell")
        self.Bind(wx.EVT_BUTTON, self.OnAboutPyShell, button)
        h_sizer.Add(button, 0, wx.ALL | wx.ALIGN_LEFT, 10)

        button = wx.Button(self, -1, "Close")
        self.Bind(wx.EVT_BUTTON, self.OnClose, button)
        h_sizer.Add(button, 0, wx.ALL | wx.ALIGN_RIGHT, 10)
        sizer.Add(h_sizer, 0)
#        sizer.Add(button, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL|wx.ALIGN_BOTTOM, 10)

        self.Center(wx.CENTER_ON_SCREEN | wx.HORIZONTAL)
        self.Fit()

    def OnClose(self, event):
        self.Close()

    def OnAboutPyShell(self, event):
        wx.py.shell.ShellFrame.OnAbout(self.Parent, event)
