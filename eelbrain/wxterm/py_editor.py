import ast
import os
import logging

import wx.stc
import wx.py
# import wx.lib.mixins.listctrl

from .. import fmtxt
from .._wxutils import droptarget, Icon, ID, key_mod


# replace Document to allow unicode i/o
from wx.py.document import Document as wxDocument
class Document(wxDocument):

    def read(self):
        text = wxDocument.read(self)
        return text

    def write(self, text):
        text = text.encode('utf-8')
        wxDocument.write(self, text)

wx.py.document.Document = Document


# fix Editor class
class Editor(wx.py.editor.Editor):

    def CallTipShow(self, pos, tip):
        self.window.CallTipShow(pos, tip)

    def GetColumn(self, pos):
        return self.window.GetColumn(pos)

wx.py.editor.Editor = Editor


class PyEditor(wx.py.editor.EditorFrame):
    """
    Editor.editor.window is based on StyledTextCtrl

    """
    def __init__(self, parent, shell, pos=(0, 0), size=(640, 840), pyfile=None,
                 name=None):
        """
        Parameters
        ----------
        pyfile : bool | str
            Filename, or True in order to display an open file dialog,
            False/None in order to open a new buffer.
        name : None | str
            Name for the new buffer.
        """
        shell.active_editor = self
        self.shell = shell
        self._exec_in_shell_namespace = True

        if isinstance(pyfile, basestring):
            filename = pyfile
        else:
            filename = None

        super(PyEditor, self).__init__(parent, title=name or "Script Editor",
                                       size=size, pos=pos, filename=filename)

    # toolbar ---
        tb = self.CreateToolBar(wx.TB_HORIZONTAL)
        tb.SetToolBitmapSize(size=(32, 32))


        # script execution
        tb.AddLabelTool(ID.EXEC_DOCUMENT, "Exec", Icon("actions/python-run"),
                        shortHelp="Execute script in shell")
        self.Bind(wx.EVT_TOOL, self.OnExecDocument, id=ID.EXEC_DOCUMENT)
        tb.AddLabelTool(ID.EXEC_SELECTION, "Exec_sel",
                        Icon("actions/python-run-selection"),
                        shortHelp="Execute selection in shell")
        self.Bind(wx.EVT_TOOL, self.OnExecSelection, id=ID.EXEC_SELECTION)
        tb.AddLabelTool(ID.EXEC_DOCUMENT_FROM_DISK, "Exec_drive",
                        Icon("actions/python-run-drive"),
                        shortHelp="Execute the whole script in the shell from "
                        "the disk. Saves unsaved changes without asking. "
                        "Warning: __file__ and sys.argv currently point to "
                        "eelbrain.__main__ instead of the executed file.")
        self.Bind(wx.EVT_TOOL, self.OnExecDocumentFromDisk,
                  id=ID.EXEC_DOCUMENT_FROM_DISK)

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
                                    size=(150, -1))
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

        tb.Realize()

    # finalize ---
        if pyfile is True:
            self.bufferOpen()
        elif not pyfile:
            self.bufferCreate()
            if name is not None:
                self.buffer.name = name

        # set icon
        if hasattr(parent, 'eelbrain_icon'):
            self.SetIcon(parent.eelbrain_icon)

        self.SetSize(size)

    def bufferCreate(self, filename=None):
        out = super(PyEditor, self).bufferCreate(filename=filename)
        self.shell.ApplyStyleTo(self.editor.window)

        zoom = self.shell.shell.GetZoom()
        self.editor.window.SetZoom(zoom)

        droptarget.set_for_strings(self.editor.window)
        self.editor.window.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
        if filename:
            self.updateTitle()

        return out

    def bufferSave(self):
        """subclassed to catch and display errors

        Returns
        -------
        cancel : bool
            True if the save operation was canceled, False otherwise.
        """
        try:
            cancel = super(PyEditor, self).bufferSave()
        except Exception as exception:
            msg = str(exception) + "\n\n(Press OK and check console for full Exception)"
            dlg = wx.MessageDialog(self,
                                   msg,
                                   "Error saving File",
                                   wx.OK | wx.CENTER | wx.ICON_ERROR)
            dlg.ShowModal()
            raise

        self.updateTitle()
        return cancel

    def bufferSaveAs(self):
        "subclass to catch and display errors"
        try:
            if not hasattr(self.buffer, 'confirmed'):
                # this caused an exception when saving a new document an
                # replacing an existing file
                self.buffer.confirmed = True

            cancel = super(PyEditor, self).bufferSaveAs()
        except Exception as exception:
            msg = str(exception) + "\n\n(Check console for more information)"
            dlg = wx.MessageDialog(self,
                                   msg,
                                   "Error saving File",
                                   wx.OK | wx.CENTER | wx.ICON_ERROR)
            dlg.ShowModal()
            raise

        self.updateTitle()
        return cancel

    def bufferSuggestSave(self):
        """Suggest saving changes.  Return True if user selected Cancel."""
        self.Raise()
        return wx.py.editor.EditorFrame.bufferSuggestSave(self)

    def bufferOpen(self):
        """Subclassed from EditorFrame."""
        cancel = wx.py.editor.EditorFrame.bufferOpen(self)
        if not cancel:
            zoom = self.shell.shell.GetZoom()
            self.editor.window.SetZoom(zoom)
        return cancel

#        cancel = wx.py.editor.EditorFrame.bufferClose(self)
# #        if cancel:
# #            return cancel
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
        Update Information in status bar. Replaces from baseclass to add
        position info and change order.
        """
        if self.editor and hasattr(self.editor, 'getStatus'):
            temp = 'Line: %d  |  Column: %d  |  Pos: %d  |  File: %s'
            fpath, line, col = self.editor.getStatus()
            if not self.editor.buffer.doc.filepath:
                fpath = "* unsaved"
            pos = self.editor.window.GetCurrentPos()
            text = temp % (line, col, pos, fpath)
            # end mod
        else:
            text = self._defaultText
        if text != self._statusText:
            self.SetStatusText(text)
            self._statusText = text

    def Destroy(self):
        logging.debug("WxTerm PyEditor.Destroy()")
        self.shell.RemovePyEditor(self)
        super(PyEditor, self).Destroy()

    def Duplicate(self):
        if not self.editor:
            return

        win = self.editor.window
        start_pos, end_pos = win.GetSelection()
        if start_pos == end_pos:  # use current line
            txt, _ = win.GetCurLine()
        else:
            txt = win.GetSelectedText()

        txt = fmtxt.unindent(txt)  # remove leading whitespaces
        self.shell.InsertStrToShell(txt)
        self.shell.Raise()

    def OnActivate(self, event=None):
        # logging.debug(" Shell Activate Event: {0}".format(event.Active))
        if not self.editor:
            return

        if event.Active:
            self.editor.window.SetCaretForeground(wx.Colour(0, 0, 0))
            self.editor.window.SetCaretPeriod(500)
            self.shell.active_editor = self
            if self._exec_in_shell_namespace:
                self.updateNamespace()
        else:
            self.editor.window.SetCaretForeground(wx.Colour(200, 200, 200))
            self.editor.window.SetCaretPeriod(0)

    def CanExec(self):
        return bool(self.editor)

    def Comment(self):
        w = self.editor.window
        start_pos, end_pos = w.GetSelection()
        start = w.LineFromPosition(start_pos)
        end = w.LineFromPosition(end_pos)
        if start > end:
            start, end = end, start
        if start != end and end_pos == w.PositionFromLine(end):
            end -= 1  # if no element in the last line is selected
        lines = range(start, end + 1)
        first = w.GetLine(lines[0])
        if len(first) == 0 or first[0] != '#':  # de-activate
            for i in lines:
                line = w.GetLine(i)
                pos = w.PositionFromLine(i)
                if len(line) > 1 and line[:2] == '##':
                    pass
                elif len(line) > 0 and line[0] == '#':
                    w.InsertText(pos, '#')
                else:
                    w.InsertText(pos, '##')
        else:  # un-comment
            for i in lines:
                while w.GetLine(i)[0] == '#':
                    pos = w.PositionFromLine(i) + 1
                    w.SetSelection(pos, pos)
                    w.DeleteBack()
        start = w.PositionFromLine(start)
        end = w.PositionFromLine(end + 1) - 1
        w.SetSelection(start, end)

    def ExecDocument(self):
        "Execute the whole script in the shell."
        txt = self.editor.getText()

        shell_globals = self._exec_in_shell_namespace
        self.shell.ExecText(txt.encode('utf-8'),
                            title=self.editor.getStatus()[0],
                            comment=None,
                            shell_globals=shell_globals,
                            filepath=self.buffer.doc.filepath,
                            internal_call=True)
        self.updateNamespace()

    def ExecDocumentFromDisk(self):
        "Save unsaved changes and execute file in shell."
        if self.bufferHasChanged():
            if self.bufferSave():
                return  # user canceled
        shell_globals = self._exec_in_shell_namespace
        self.shell.ExecFile(self.buffer.doc.filepath,
                            shell_globals=shell_globals)
        self.updateNamespace()

    def ExecSelection(self):
        """Execute the current selection, or the current line if nothing is
        selected.
        """
        win = self.editor.window
        start_pos, end_pos = win.GetSelection()
        if start_pos == end_pos:  # run current line
            # extended selection including hierarchically subordinate lines
            i_first = win.GetCurrentLine()
            indent = win.GetLineIndentation(i_first)
            i_last = i_first
            n_lines = win.GetLineCount()
            while True:
                if i_last + 1 >= n_lines:
                    break
                elif win.GetLineIndentation(i_last + 1) > indent:
                    pass
                elif not win.GetLine(i_last + 1).strip():
                    pass
                else:
                    start_pos = win.PositionFromLine(i_first)
                    end_pos = win.GetLineEndPosition(i_last)
                    code = win.GetTextRange(start_pos, end_pos)
                    code = fmtxt.unindent(code)
                    try:
                        ast.parse(code, mode='single')
                    except SyntaxError:
                        pass
                    else:
                        break
                i_last += 1

            start_pos = win.PositionFromLine(i_first)
            end_pos = win.GetLineEndPosition(i_last)

            # execute the code
            cmd = win.GetTextRange(start_pos, end_pos)
            if not cmd.strip():
                return
            self.shell.ExecCommand(cmd)

            # move the caret to the next line
            win.SetSelection(end_pos, end_pos)
            self.MoveCaretToNextLine()
        else:  # execute the selection
            txt = self.editor.window.GetSelectedText()
            if not txt:
                return

            # fix selection
            start_pos, end_pos = win.GetSelection()
            if start_pos > end_pos:
                start_pos, end_pos = end_pos, start_pos

            start = win.LineFromPosition(start_pos + 1)
            end = win.LineFromPosition(end_pos)
            if start == end:
                comment = "line %s" % start
            else:
                comment = "lines %s - %s" % (start, end)

            shell_globals = self._exec_in_shell_namespace
            self.shell.ExecText(txt,
                                title=self.editor.getStatus()[0],
                                comment=comment,
                                shell_globals=shell_globals,
                                filepath=self.buffer.doc.filepath,
                                internal_call=True)
        self.updateNamespace()

    def MoveCaretToNextLine(self):
        """Move the caret to the next non-empty line

        Returns
        -------
        pos : int
            New caret position.
        """
        win = self.editor.window
        cur_line = win.GetCurrentLine()
        if cur_line == win.GetLineCount() - 1:
            pos = win.PositionFromLine(cur_line)
        else:
            next_line = cur_line + 1
            while True:
                if next_line == win.GetLineCount() - 1:
                    break
                else:
                    line = win.GetLine(next_line).strip()
                    if line and not line.startswith('#'):
                        break
                    else:
                        next_line += 1

            pos = win.PositionFromLine(next_line)

        win.SetSelection(pos, pos)
        return pos

    def OnExecDocument(self, event=None):
        if self.CanExec():
            self.ExecDocument()

    def OnExecDocumentFromDisk(self, event=None):
        if self.CanExec():
            self.ExecDocumentFromDisk()

    def OnExecSelection(self, event=None):
        "Execute the selection in the shell."
        if self.CanExec():
            self.ExecSelection()

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
        self.ToolBar.SetToolNormalBitmap(ID.PYDOC_EXEC_ISOLATE, bmp)
        self.updateNamespace()

    def GetCurLine(self):
        return self.editor.window.GetCurLine()

    def OnKeyDown(self, event):
        # these ought to be handles in stc.StyledTextCtrl
        # src/osx_cocoa/stc.py ?
        key = event.GetKeyCode()
        mod = key_mod(event)
        if any(mod):
            mod_str = '-'.join(d for d, s in zip(['ctrl', 'cmd', 'alt'], mod) if s)
            logging.info("Editor OnKeyDown: {0} {1}".format(mod_str, key))

        if mod == [1, 0, 0]:  # [ctrl]
            event.Skip()
        elif mod == [0, 0, 1]:  # alt down
            if key in [315, 317]:  # arrow
                # FIXME: when last line without endline is selected, someting
                #        bad happens
                w = self.editor.window
                w.LineCut()
                if key == 315:  # [up]
                    w.LineUp()
                elif key == 317:  # [down]
                    w.LineDown()
                w.Paste()
                w.LineUp()
                return
            else:
                event.Skip()
#        elif mod == [0, 1, 0]: # command (Os-X)
#            pass
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
        self.updateNamespace()

    # Text Search
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

    def updateNamespace(self):
        if self._exec_in_shell_namespace:
            self.buffer.interp.locals.clear()
            self.buffer.interp.locals.update(self.shell.global_namespace)
        else:
            super(Editor, self).updateNamespace()
#        self.buffer.updateNamespace()

    def updateTitle(self):
        if self.editor is None:
            self.SetTitle("Empty Editor")
        else:
            path = self.editor.getStatus()[0]
            # add to recent paths
            self.shell.recent_item_add(path, 'pydoc')
            # set title
            name = os.path.basename(path)
            self.SetTitle("Script Editor: %s" % name)

