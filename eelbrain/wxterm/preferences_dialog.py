'''
Created on Nov 8, 2011

@author: christian
'''
import os, wx

Help_dataDir = ("DataDir is used by the pyShell. A file called 'startup' in the dataDir is "
                "executed as startup script (restart Eelbrain for changes to take effect)")



class PreferencesDialog(wx.Frame):
    def __init__(self, shell, ID=wx.ID_ANY, pos=wx.DefaultPosition,
                 size=(500,50), style=wx.DEFAULT_FRAME_STYLE): #wx.DefaultSize
        """
        tutorial on layout:
        http://zetcode.com/wxpython/layout/
        
        """
        title = "Eelbrain Preferences"
        wx.Frame.__init__(self, shell, ID, title, pos, size, style)
        self.config = shell.wx_config
        
#        pref_sizer = wx.BoxSizer(wx.VERTICAL)
        
    # Data Dir
        panel_dataDir = wx.Panel(self, -1)#, size=(500,300))
        panel_dataDir.SetBackgroundColour("BLUE")
        dataDir = self.config.Read("dataDir")
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        ID_DATADIR = wx.NewId()
        button = wx.Button(panel_dataDir, ID_DATADIR, "set dataDir")
        self.Bind(wx.EVT_BUTTON, self.SetDataDir, id=ID_DATADIR)
        sizer.Add(button, 0, wx.ALIGN_LEFT|wx.EXPAND)
        # path
        txt = self.dataDirTxt = wx.TextCtrl(panel_dataDir, -1, dataDir, style=wx.TE_READONLY)#, size=(500,100))
        sizer.Add(txt, 1, wx.EXPAND|wx.ALIGN_RIGHT)
        # edit startup script
        ID_EDIT = wx.NewId()
        button = wx.Button(panel_dataDir, ID_EDIT, "Edit Startup Script")
        self.Bind(wx.EVT_BUTTON, self.EditStartupScript, id=ID_EDIT)
        sizer.Add(button, 0, wx.ALIGN_LEFT|wx.EXPAND)        
        # help btn
        btn = wx.Button(panel_dataDir, wx.ID_HELP)
        sizer.Add(btn, 0, wx.ALIGN_RIGHT)
        self.Bind(wx.EVT_BUTTON, self.OnHelpDataDir, btn)
        
        # 
        panel_dataDir.SetSizer(sizer)
#        pref_sizer.Add(sizer, 0)
        
#        panel_dataDir.Fit()
        
#        self.SetSizer(pref_sizer)
#        self.Fit()
#        self.SetAutoLayout(True)
#        self.Layout()
    def OnHelpDataDir(self, event=None):
        dlg = wx.MessageDialog(self, Help_dataDir, "Help: dataDir", 
                               wx.OK|wx.ICON_INFORMATION)
        dlg.ShowModal()
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

        
        