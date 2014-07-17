import wx

from .._wxutils import ID, logger


class App(wx.App):
    def OnInit(self):
        self.SetExitOnFrameDelete(False)
        self.SetAppName("Eelbrain")

        # File Menu
        m = file_menu = wx.Menu()
        m.Append(wx.ID_CLOSE, '&Close Window \tCtrl+W')
        m.Append(wx.ID_SAVE, "Save \tCtrl+S")
        m.Append(wx.ID_SAVEAS, "Save As... \tCtrl+Shift+S")

        # Edit Menu
        m = edit_menu = wx.Menu()
        m.Append(ID.UNDO, '&Undo \tCtrl+Z')
        m.Append(ID.REDO, '&Redo \tCtrl+Shift+Z')
        m.AppendSeparator()
        m.Append(wx.ID_COPY, '&Copy \tCtrl+C')
        m.AppendSeparator()
        m.Append(wx.ID_CLEAR, 'Cle&ar')

        # Menu Bar
        menu_bar = wx.MenuBar()
        menu_bar.Append(file_menu, "File")
        menu_bar.Append(edit_menu, "Edit")
        wx.MenuBar.MacSetCommonMenuBar(menu_bar)

        # Bind Menu Commands
        self.Bind(wx.EVT_MENU, self.OnClear, id=wx.ID_CLEAR)
        self.Bind(wx.EVT_MENU, self.OnCloseWindow, id=wx.ID_CLOSE)
        self.Bind(wx.EVT_MENU, self.OnCopy, id=wx.ID_COPY)
        self.Bind(wx.EVT_MENU, self.OnRedo, id=ID.REDO)
        self.Bind(wx.EVT_MENU, self.OnSave, id=wx.ID_SAVE)
        self.Bind(wx.EVT_MENU, self.OnSaveAs, id=wx.ID_SAVEAS)
        self.Bind(wx.EVT_MENU, self.OnUndo, id=ID.UNDO)

        # bind update UI
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUIClear, id=wx.ID_CLEAR)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUIClose, id=wx.ID_CLOSE)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUICopy, id=wx.ID_COPY)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUIRedo, id=ID.REDO)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUISave, id=wx.ID_SAVE)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUISaveAs, id=wx.ID_SAVEAS)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUIUndo, id=ID.UNDO)

        return True

    def _get_active_frame(self):
        win = wx.Window.FindFocus()
        return wx.GetTopLevelParent(win)
        for w in  wx.GetTopLevelWindows():
            if hasattr(w, 'IsActive') and w.IsActive():
                return w

    def OnClear(self, event):
        frame = self._get_active_frame()
        frame.Clear()

    def OnCloseWindow(self, event):
        frame = self._get_active_frame()
        logger.debug("Close %r" % frame)
        if frame is not None:
            frame.Close()

    def OnCopy(self, event):
        win = wx.Window.FindFocus()
        logger.debug("Copy %r" % win)
        win.Copy()

    def OnRedo(self, event):
        frame = self._get_active_frame()
        frame.Redo()

    def OnSave(self, event):
        frame = self._get_active_frame()
        frame.Save()

    def OnSaveAs(self, event):
        frame = self._get_active_frame()
        frame.SaveAs()

    def OnUndo(self, event):
        frame = self._get_active_frame()
        frame.Undo()

    def OnUpdateUIClear(self, event):
        frame = self._get_active_frame()
        if frame is None:
            event.Enable(False)
        else:
            frame.OnUpdateUIClear(event)

    def OnUpdateUIClose(self, event):
        frame = self._get_active_frame()
        if frame is None:
            event.Enable(False)
        else:
            frame.OnUpdateUIClose(event)

    def OnUpdateUICopy(self, event):
        frame = self._get_active_frame()
        if frame is None:
            event.Enable(False)
        else:
            frame.OnUpdateUICopy(event)

    def OnUpdateUIRedo(self, event):
        frame = self._get_active_frame()
        if frame is None:
            event.Enable(False)
        else:
            frame.OnUpdateUIRedo(event)

    def OnUpdateUISave(self, event):
        frame = self._get_active_frame()
        if frame is None:
            event.Enable(False)
        else:
            frame.OnUpdateUISave(event)

    def OnUpdateUISaveAs(self, event):
        frame = self._get_active_frame()
        if frame is None:
            event.Enable(False)
        else:
            frame.OnUpdateUISaveAs(event)

    def OnUpdateUIUndo(self, event):
        frame = self._get_active_frame()
        if frame is None:
            event.Enable(False)
        else:
            frame.OnUpdateUIUndo(event)


def get_app():
    app = wx.GetApp()
    if app is None:
        logger.debug("No wx.App found, initializing Eelbrain App")
        app = App()
    return app


def run():
    app = get_app()
    if app.IsMainLoopRunning():
        raise RuntimeError("MainLoop is already running")
    else:
        app.MainLoop()
