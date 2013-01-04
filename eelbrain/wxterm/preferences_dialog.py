'''
Created on Nov 8, 2011

@author: christian
'''
import logging
import os

import wx

import ID

Help_dataDir = ("DataDir is used by the pyShell. A file called 'startup' in the dataDir is "
                "executed as startup script (restart Eelbrain for changes to take effect)")



class PreferencesDialog(wx.Frame):
    def __init__(self, shell, Id=wx.ID_ANY, pos=wx.DefaultPosition,
                 size=(500, 50), style=wx.DEFAULT_FRAME_STYLE):  # wx.DefaultSize
        """
        tutorial on layout:
        http://zetcode.com/wxpython/layout/

        """
        title = "Eelbrain Preferences"
        wx.Frame.__init__(self, shell, Id, title, pos, size, style)
        self.config = shell.wx_config

        pref_sizer = wx.BoxSizer(wx.VERTICAL)

    # Data Dir ---
        panel_dataDir = wx.Panel(self, -1)  # , size=(500,300))
        panel_dataDir.SetBackgroundColour("BLUE")
        dataDir = self.config.Read("dataDir")
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        ID_DATADIR = wx.NewId()
        button = wx.Button(panel_dataDir, ID_DATADIR, "set dataDir")
        self.Bind(wx.EVT_BUTTON, self.SetDataDir, id=ID_DATADIR)
        sizer.Add(button, 0, wx.ALIGN_LEFT | wx.EXPAND)
        # path
        txt = self.dataDirTxt = wx.TextCtrl(panel_dataDir, -1, dataDir,
                                            size=(400, 0), style=wx.TE_READONLY)
        sizer.Add(txt, 1, wx.EXPAND | wx.ALIGN_RIGHT)
        # edit startup script
        ID_EDIT = wx.NewId()
        button = wx.Button(panel_dataDir, ID_EDIT, "Edit Startup Script")
        self.Bind(wx.EVT_BUTTON, self.EditStartupScript, id=ID_EDIT)
        sizer.Add(button, 0, wx.ALIGN_LEFT | wx.EXPAND)
        # help btn
        btn = wx.Button(panel_dataDir, wx.ID_HELP)
        sizer.Add(btn, 0, wx.ALIGN_RIGHT)
        self.Bind(wx.EVT_BUTTON, self.OnHelpDataDir, btn)

        pref_sizer.Add(sizer, 0)


        # Font ---
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        button = wx.Button(panel_dataDir, ID.SET_FONT, "Font")
        self.Bind(wx.EVT_BUTTON, self.OnSetFont, id=ID.SET_FONT)
        sizer.Add(button, 0, wx.ALIGN_LEFT | wx.EXPAND)
        pref_sizer.Add(sizer, 0)

        panel_dataDir.SetSizer(pref_sizer)

        pref_sizer.Fit(self)

    def OnHelpDataDir(self, event=None):
        dlg = wx.MessageDialog(self, Help_dataDir, "Help: dataDir",
                               wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()

    def OnSetFont(self, event):
        try:
            cur_size = int(self.config.Read('font size'))
        except:
            cur_size = 12

        font = wx.Font(pointSize=cur_size,
                       family=wx.FONTFAMILY_UNKNOWN,
                       style=wx.FONTSTYLE_NORMAL,
                       weight=wx.FONTWEIGHT_NORMAL,
                       face=self.config.Read('font'))
        data = wx.FontData()
        data.EnableEffects(True)
        data.SetInitialFont(font)
        data.SetColour(self.config.Read('font color'))
        dlg = wx.FontDialog(self, data)
        if dlg.ShowModal() == wx.ID_OK:
            data = dlg.GetFontData()
            font_ = data.GetChosenFont()
            font = font_.GetFaceName()
            size = font_.GetPointSize()
            color_ = data.GetColour()
            color = color_.GetAsString(wx.C2S_HTML_SYNTAX)

            logging.debug('You selected: "%s", %d points, color %s\n' %
                          (font, size, color))

            self.config.Write("font", font)
            self.config.Write("font size", str(size))
            self.config.Write("font color", color)
            self.Parent.ApplyStyle()

        dlg.Destroy()

    def SetDataDir(self, event=None):
        dlg = wx.DirDialog(self, "Select user dataDir directory")
        if dlg.ShowModal() == wx.ID_OK:
            dataDir = dlg.GetPath()
            self.config.Write("dataDir", dataDir)
            self.dataDirTxt.SetValue(dataDir)
        dlg.Destroy()

    def EditStartupScript(self, event=None):
        dataDir = self.config.Read("dataDir")
        path = os.path.join(dataDir, 'startup')
        if not os.path.exists(path):
            with open(path, 'w') as f:
                f.write("# Eelbrain startup script")
        self.Parent.create_py_editor(pyfile=path)



