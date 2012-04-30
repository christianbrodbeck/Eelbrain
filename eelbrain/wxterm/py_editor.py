import os 
import logging

import wx.stc
import wx.py
#import wx.lib.mixins.listctrl

import ID
from eelbrain import wxutils
from eelbrain.wxutils import Icon



class Editor(wx.py.editor.EditorFrame):
    """
    Editor.editor.window is based on StyledTextCtrl
    
    """
    def __init__(self, parent, shell, pos=(0,0), pyfile=None):
        """
        :arg pyfile: filename as string, or True in order to display an open
            file dialog
         
        """
        self.shell = shell
        y_size = min([wx.Display().GetGeometry()[-1] - 20, 840])
        
        if isinstance(pyfile, basestring):
            filename = pyfile
        else:
            filename = None
        
        wx.py.editor.EditorFrame.__init__(self, parent, title="Script Editor", 
                                          size=(640, y_size), pos=pos,
                                          filename=filename)
        
        if pyfile is True:
            self.bufferOpen()
        elif not pyfile:
            self.bufferCreate()
        
        shell.active_editor = self # for ctrl-'/' insertion
        self._exec_in_shell_namespace = True
        
        
    ############################################
    # toolbar
        tb = self.toolbar = self.CreateToolBar(wx.TB_HORIZONTAL)
        tb.SetToolBitmapSize(size=(32,32))
        
        
        # script execution
        tb.AddLabelTool(ID.PYDOC_EXEC, "Exec", Icon("actions/python-run"),
                        shortHelp="Execute script in shell")
        self.Bind(wx.EVT_TOOL, self.OnExec, id=ID.PYDOC_EXEC)
        tb.AddLabelTool(ID.PYDOC_EXEC_SEL, "Exec_sel", Icon("actions/python-run-selection"),
                        shortHelp="Execute selection in shell")
        self.Bind(wx.EVT_TOOL, self.OnExecSelected, id=ID.PYDOC_EXEC_SEL)
        tb.AddLabelTool(ID.PYDOC_EXEC_DRIVE, "Exec_drive", 
                        Icon("actions/python-run-drive"),
                        shortHelp="Execute the whole script in the shell from "
                                  "the disk. Saves unsaved changes without "
                                  "asking. Warning: __file__ and sys.argv cur"
                                  "rently point to eelbrain.__main__ instead of"
                                  " the executed file.")
        self.Bind(wx.EVT_TOOL, self.OnExecFromDrive, id=ID.PYDOC_EXEC_DRIVE)
        
        if self._exec_in_shell_namespace:
            icon = Icon("actions/terminal-on")
        else:
            icon = Icon("actions/terminal-off")
        tb.AddLabelTool(ID.PYDOC_EXEC_ISOLATE, "Exec-Globals", icon,
                        shortHelp="Toggle script execution in the shell's name"
                                  "space (icon terminal on) or in a separate "
                                  "namespace (icon terminal off).")
        self.Bind(wx.EVT_TOOL, self.OnExec_ToggleIsolate, id=ID.PYDOC_EXEC_ISOLATE)
        
        tb.AddLabelTool(wx.ID_REFRESH, "Update Namespace", 
                        Icon("tango/actions/view-refresh"),
                        shortHelp="Refresh local namespace from script (for "
                                  "auto-completion)")
        self.Bind(wx.EVT_TOOL, self.OnUpdateNamespace, id=wx.ID_REFRESH)
        tb.AddSeparator()
        
        
        # file operations
        tb.AddLabelTool(wx.ID_NEW, "New", Icon("documents/pydoc-new"), 
                        shortHelp="New Document")
        tb.AddLabelTool(wx.ID_OPEN, "Load", Icon("documents/pydoc-open"),
                        shortHelp="Open Document")
        tb.AddLabelTool(wx.ID_SAVEAS, "Save As", 
                        Icon("tango/actions/document-save-as"),
                        shortHelp="Save Document as ...")
        tb.AddLabelTool(wx.ID_SAVE, "Save", Icon("tango/actions/document-save"),
                        shortHelp="Save Document")
        tb.AddSeparator()
        
#        self.Bind(wx.EVT_MENU_CLOSE, self.OnFileClose)
#        self.Bind(wx.EVT_MENU, self.OnClose, id=wx.ID_CLOSE)
        
        
        # text search
        self._search_string = None
        search_ctrl = wx.SearchCtrl(tb, id=wx.ID_HELP, style=wx.TE_PROCESS_ENTER, 
                                    size=(150,-1))
        search_ctrl.Bind(wx.EVT_TEXT, self.OnSearchMod)
        search_ctrl.Bind(wx.EVT_TEXT_ENTER, self.OnSearchForward)
        search_ctrl.ShowCancelButton(True)
        self.Bind(wx.EVT_SEARCHCTRL_CANCEL_BTN, self.OnSearchCancel, search_ctrl)
        search_ctrl.ShowSearchButton(True)
        self.Bind(wx.EVT_SEARCHCTRL_SEARCH_BTN, self.OnSearchForward, search_ctrl)
#        search_ctrl.SetMenu(self.search_history_menu)
#        self.search_history_menu.Bind(wx.EVT_MENU, self.OnSearchHistory) 
        tb.AddControl(search_ctrl)
        self.search_ctrl = search_ctrl
        self.search_str = ''
        self.search_flag = wx.stc.STC_FIND_REGEXP
        
        tb.AddLabelTool(wx.ID_BACKWARD, "Backward", Icon("tango/actions/go-previous"),
                        shortHelp="Search backwards")
        self.Bind(wx.EVT_TOOL, self.OnSearchBackward, id=wx.ID_BACKWARD)
        tb.AddLabelTool(wx.ID_FORWARD, "Forward", Icon("tango/actions/go-next"),
                        shortHelp="Search forward")
        self.Bind(wx.EVT_TOOL, self.OnSearchForward, id=wx.ID_FORWARD)
        
        # set icon
        if hasattr(parent, 'eelbrain_icon'):
            self.SetIcon(parent.eelbrain_icon)
#        tb.AddLabelTool(ID.PYDOC_HASH, "Comment", Icon("mine.#"))
        tb.Realize()
        
        # if toolbar is not entirely visible, increase window width
        tb_width = tb.Size[0]
        size = self.GetSize()
        diff = tb_width - size[0]
        if diff > 0:
            size[0] = size[0] + diff
            pos = self.GetPosition()
            pos[0] = max(0, pos[0]-diff)
            self.SetSize(size)
            self.SetPosition(pos)
        else:
            size[1] -= 2
            self.SetSize(size)
    
    def bufferCreate(self, filename=None):
        wx.py.editor.EditorFrame.bufferCreate(self, filename=filename)
        self.shell.SetColours(self.editor.window)
        
        zoom = self.shell.shell.GetZoom()
        self.editor.window.SetZoom(zoom)
        
        wxutils.droptarget.set_for_strings(self.editor.window)
        self.editor.window.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
        if filename:
            self.updateTitle()
    
    def bufferSave(self):
        "subclassed to catch and display errors"
        try:
            wx.py.editor.EditorFrame.bufferSave(self)
        except Exception as exception:
            msg = str(exception) + "\n\n(Press OK and check console for full Exception)"
            dlg = wx.MessageDialog(self, 
                                   msg,
                                   "Error saving File",
                                   wx.OK|wx.CENTER|wx.ICON_ERROR)
            dlg.ShowModal()
            raise
        else:
            self.updateTitle()
    
    def bufferSaveAs(self):
        "subclass to catch and display errors"
        try:
            wx.py.editor.EditorFrame.bufferSaveAs(self)
        except Exception as exception:
            msg = str(exception) + "\n\n(Check console for more information)"
            dlg = wx.MessageDialog(self, 
                                   msg,
                                   "Error saving File",
                                   wx.OK|wx.CENTER|wx.ICON_ERROR)
            dlg.ShowModal()
            raise
        else:
            self.updateTitle()
    
    def bufferOpen(self):
        """Subclassed from EditorFrame."""
        cancel = wx.py.editor.EditorFrame.bufferOpen(self)
        if not cancel:
            zoom = self.shell.shell.GetZoom()
            self.editor.window.SetZoom(zoom)
        return cancel
    
#        cancel = wx.py.editor.EditorFrame.bufferClose(self)
##        if cancel:
##            return cancel
#        if self.bufferHasChanged():
#            cancel = self.bufferSuggestSave()
#            if cancel:
#                return cancel
#        
#        filedir = ''
#        if self.buffer and self.buffer.doc.filedir:
#            filedir = self.buffer.doc.filedir
#        
#        path = ui.ask_file(title="Open Python Script", 
#                           message="Select Python Script to load", 
#                           ext=[('py', "Python script")],
#                           directory=filedir)
#        
#        if isstr(path) and os.path.isfile(path):
#            self.bufferCreate(path)
#        else:
#            logging.info("Py editor OnFileOpen with nonexistent path: %s"%path)
#    def OnClose(self, event):
#        logging.debug('OnClose')
    def _updateStatus(self):
        """
        Update Information in status bar. Copied from baseclass except for 
        adding Pos information.
        
        """
        if self.editor and hasattr(self.editor, 'getStatus'):
            status = self.editor.getStatus()
            text = 'File: %s  |  Line: %d  |  Column: %d' % status
            # mod: adding pos info
            pos = self.editor.window.GetCurrentPos()
            text += '  |  Pos: %d' % pos
            # end mod
        else:
            text = self._defaultText
        if text != self._statusText:
            self.SetStatusText(text)
            self._statusText = text


    def Destroy(self):
        logging.debug("Editor.Destroy()")
        self.shell.RemovePyEditor(self)
        wx.py.editor.EditorFrame.Destroy(self)
    def OnActivate(self, event=None):
        #logging.debug(" Shell Activate Event: {0}".format(event.Active))
        if self.editor:
            if event.Active:
                self.editor.window.SetCaretForeground(wx.Colour(0,0,0))
                self.editor.window.SetCaretPeriod(500)
                self.shell.active_editor = self 
            else:
                self.editor.window.SetCaretForeground(wx.Colour(200,200,200))
                self.editor.window.SetCaretPeriod(0)
    def OnExec(self, event=None):
        "Execute the whole script in the shell."
        if self.editor:
            txt = self.editor.window.GetSelectedText()
            txt = self.editor.getText()
            
            shell_globals = self._exec_in_shell_namespace
            self.shell.ExecText(txt,
                                title=self.editor.getStatus()[0], 
                                comment=None,
                                shell_globals=shell_globals,
                                filedir=self.buffer.doc.filedir,
                                internal_call=True)
    def OnExecFromDrive(self, event=None):
        "Save unsaved changes and execute file in shell."
        self.bufferSave()        
        shell_globals = self._exec_in_shell_namespace
        self.shell.ExecFile(self.buffer.doc.filepath, 
                            shell_globals=shell_globals)
    def OnExecSelected(self, event=None):
        "Execute the selection in the shell."
        if self.editor:
            txt = self.editor.window.GetSelectedText()
            if txt:
                # find line numbers
                w = self.editor.window
                start_pos, end_pos = w.GetSelection()
                if start_pos > end_pos:
                    start_pos, end_pos = end_pos, start_pos
                
                start = w.LineFromPosition(start_pos + 1)
                end = w.LineFromPosition(end_pos)
                if start == end:
                    comment = "line %s"%start
                else:
                    comment = "lines %s - %s"%(start, end)
                
                shell_globals = self._exec_in_shell_namespace
                self.shell.ExecText(txt,
                                    title=self.editor.getStatus()[0], 
                                    comment=comment, 
                                    shell_globals=shell_globals,
                                    filedir=self.buffer.doc.filedir,
                                    internal_call=True)
            else:
                pass
    def OnExec_ToggleIsolate(self, event=None):
        """
        Toggle whether code is executed with the shell's globals or in a 
        separate namespace
        
        """
        self._exec_in_shell_namespace = not self._exec_in_shell_namespace
        if self._exec_in_shell_namespace:
            bmp = Icon("actions/terminal-on")
        else:
            bmp = Icon("actions/terminal-off")
        self.toolbar.SetToolNormalBitmap(ID.PYDOC_EXEC_ISOLATE, bmp)
    def GetCurLine(self):
        return self.editor.window.GetCurLine()
    def OnKeyDown(self, event):
        # these ought to be handles in stc.StyledTextCtrl
        # src/osx_cocoa/stc.py ?
        key = event.GetKeyCode()
        mod = wxutils.key_mod(event)
        if any(mod):
            logging.warning("Editor OnKeyDown: {0} {1}".format(mod, key))
        
        if mod == [1, 0, 0]: # [ctrl]
            if key == 47: # [/] --> comment
                w = self.editor.window
                start_pos, end_pos = w.GetSelection()
                start = w.LineFromPosition(start_pos)
                end = w.LineFromPosition(end_pos)
                if start > end:
                    start, end = end, start
                if start != end and end_pos == w.PositionFromLine(end):
                    end -= 1 # if no element in the last line is selected
                lines = range(start, end+1)
                first = w.GetLine(lines[0])
                if len(first) == 0 or first[0] != '#': # de-activate
                    for i in lines:
                        line = w.GetLine(i)
                        pos = w.PositionFromLine(i)
                        if len(line) > 1 and line[:2] == '##':
                            pass
                        elif len(line) > 0 and line[0] == '#':
                            w.InsertText(pos, '#')
                        else:
                            w.InsertText(pos, '##')
                else: # un-comment
                    for i in lines:
                        while w.GetLine(i)[0] == '#':
                            pos = w.PositionFromLine(i) + 1
                            w.SetSelection(pos, pos)
#                                w.SetCurrentPos(pos)
                            w.DeleteBack()
                start = w.PositionFromLine(start)
                end = w.PositionFromLine(end+1)-1
                w.SetSelection(start, end)
                return
            elif key == 82: # r -> execute the script
                self.OnExecFromDrive(event)
            elif key == 69: # e -> execute excerpt
                self.OnExecSelected(event)
        elif mod == [0, 0, 1]: # alt down
            if key in [315, 317]: # arrow
                # FIXME: when last line without endline is selected, someting 
                #        bad happens
                w = self.editor.window
                w.LineCut()
                if key == 315: # [up]
                    w.LineUp()
                elif key == 317: # [down]
                    w.LineDown()
                w.Paste()
                w.LineUp()
                return
        elif mod == [0, 1, 0]: # command (Os-X)
            pass
        else:
            event.Skip()
    
    def InsertLine(self, line):
        w = self.editor.window
        
        cur_line, cur_line_pos = w.GetCurLine()
        if cur_line_pos == 0:
            pos = w.GetCurrentPos()
            w.InsertText(pos, line)
            newpos = pos + len(line)
            w.SetSelection(newpos, newpos)
            w.NewLine()
        else:
            w.LineEnd()
#            w.NewLine()
            pos = w.GetCurrentPos() + 1
            w.SetSelection(pos, pos)
            w.InsertText(pos, line)
        
    def OnUpdateNamespace(self, event=None):
        self.buffer.updateNamespace()
    
    ## Text Search
    def OnSearchMod(self, event):
        self.search_str = event.GetString()
        self.OnSearchForward(event)
    def OnSearchForward(self, event):
        w = self.editor.window
        old_pos = w.GetSelection()
        new_pos = max(old_pos) + 1
        w.SetSelection(new_pos, new_pos)
        w.SearchAnchor()
        index = w.SearchNext(self.search_flag, self.search_str)
        if index == -1:
            w.SetSelection(min(old_pos), max(old_pos))
        else:
            w.ScrollToLine(w.LineFromPosition(index) - 15)
    def OnSearchBackward(self, event):
        w = self.editor.window
        old_pos = w.GetSelection()
        new_pos = min(old_pos) - 1
        w.SetSelection(new_pos, new_pos)
        w.SearchAnchor()
        index = w.SearchPrev(self.search_flag, self.search_str)
        if index == -1:
            w.SetSelection(min(old_pos), max(old_pos))
        else:
            w.ScrollToLine(w.LineFromPosition(index) - 15)
    def OnSearchCancel(self, event=None):
        self.search_ctrl.Clear()
        self._search_string = None
    
    def updateTitle(self):
        if self.editor is None:
            self.SetTitle("Empty Editor")
        else:
            path = self.editor.getStatus()[0]
            # add to recent paths
            self.shell.recent_item_add(path, 'pydoc')
            # set title
            name = os.path.basename(path)
            self.SetTitle("Script Editor: %s"%name)

