"""
This module provides a simple user interface for tables. Tables can be sent
to the shell, saved as tab separated value (TSV) files and pickled.

Classes
-------
TableFrame is the frame which contains the grid
MyGrid_simple is the grid
TableAttachDialog asks for output options

"""

import logging
import cPickle as pickle

import numpy as np
import wx
import wx.grid
# import wx.lib.mixins.grid

from .._wxutils import Icon, ID
from .._utils.basic import add_ext, loadtable



class MyGrid_simple(wx.grid.Grid):  # , wx.lib.mixins.grid.GridAutoEditMixin): #wx.grid.PyGridTableBase,
    def __init__(self, parent):
        wx.grid.Grid.__init__(self, parent)
        self.parent = parent
        # wx.grid.PyGridTableBase.__init__(self)
        # wx.lib.mixins.grid.GridAutoEditMixin.__init__(self)
        self.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
    def OnKeyDown(self, event=None):
        key = event.GetKeyCode()
        logging.debug("MyGrid_simple.OnKeyDown: %s" % key)
        if key == wx.WXK_TAB:
            if self.GetGridCursorCol() == self.GetNumberCols() - 1:
                y = self.GetGridCursorRow() + 1
                if y == self.GetNumberRows():
                    self.parent.AppendRow()
                self.SetGridCursor(y, 0)
                return
        elif key in [wx.WXK_BACK, wx.WXK_DELETE]:
            cols, rows = self.GetSelectedCols(), self.GetSelectedRows()
            for c in cols:
                self.DeleteCols(c, 1)
            for r in rows:
                self.DeleteRows(r, 1)
            if rows or cols:
                return  # do not select cell content if row/col is deleted
        event.Skip()
    def set(self, table):
        x = max(len(row) for row in table)
        y = len(table)
        # self.ClearGrid() #NOTENOUGH
        self.parent.SetColsRows(x, y)
        for j, row in enumerate(table):
            for i, item in enumerate(row):
                self.SetCellValue(j, i, unicode(item))
        # self.AutoSize()
    def load(self, path):
        table = loadtable(path)
        self.set(table)
    def get(self, dt=unicode,
            asdict=False, kdt=unicode):
        x = self.GetNumberCols()
        y = self.GetNumberRows()
        if asdict == False:
            table = []
            for i in range(y):
                row = []
                for j in range(x):
                    value = self.GetCellValue(i, j)
                    if value in ['', 'nan', 'NAN']:
                        value = np.NAN
                    else:
                        try:
                            value = dt(value)
                        except:
                            logging.warning(" MyGrid_simple could not convert %s to %s" % (value, dt))
                            value = np.NAN
                    row.append(value)
                table.append(row)
        else:
            table = {}
            if asdict == 'row':
                for j in range(x):
                    col = []
                    for i in range(1, y):
                        col.append(dt(self.GetCellValue(i, j)))
                    if len(col) == 1:
                        col = col[0]
                    table[kdt(self.GetCellValue(0, j))] = col
            elif asdict == 'col':
                for i in range(y):
                    row = []
                    for j in range(1, x):
                        row.append(dt(self.GetCellValue(i, j)))
                    if len(row) == 1:
                        row = row[0]
                    table[kdt(self.GetCellValue(i, 0))] = row
            else:
                raise NotImplementedError("asdict needs to be False, 'row', or 'col'")
        return table
    def resize(self, cols, rows):
        cols_now = self.GetNumberCols()
        if cols_now > cols:
            pass
        else:
            self.AppendCols(cols - cols_now)
        rows_now = self.GetNumberRows()
        if rows_now > rows:
            pass
        else:
            self.AppendRows(rows - rows_now)


class TableAttachDialog(wx.Dialog):
    """
    Dialog to set options for attaching table

    """
    dtypes_s = ['int', 'float', 'unicode']
    dtypes = [int, float, unicode]
    # built on wxPython Demo 'Dialog' Example
    def __init__(self, parent, default_name,
                 title="Attach Table",
                 message="Settings for attaching table to global namespace",
                 ID=-1, size=wx.DefaultSize,
                 pos=wx.DefaultPosition, style=wx.DEFAULT_DIALOG_STYLE,
                 useMetal=False,):
        # Instead of calling wx.Dialog.__init__ we precreate the dialog
        # so we can set an extra style that must be set before
        # creation, and then we create the GUI object using the Create
        # method.
        pre = wx.PreDialog()
        pre.SetExtraStyle(wx.DIALOG_EX_CONTEXTHELP)
        pre.Create(parent, ID, title, pos, size, style)

        # This next step is the most important, it turns this Python
        # object into the real wrapper of the dialog (instead of pre)
        # as far as the wxPython extension is concerned.
        self.PostCreate(pre)

        # This extra style can be set after the UI object has been created.
        if 'wxMac' in wx.PlatformInfo and useMetal:
            self.SetExtraStyle(wx.DIALOG_EX_METAL)


        # Now continue with the normal construction of the dialog
        # contents
        sizer = wx.BoxSizer(wx.VERTICAL)

        label = wx.StaticText(self, -1, message)
        sizer.Add(label, 0, wx.ALIGN_CENTRE | wx.ALL, 5)

        ## CONTROLS #####   #####   #####   #####   #####   #####

        rb = self.dt_ctrl = wx.RadioBox(self, -1, "dtype",
                                  wx.DefaultPosition, wx.DefaultSize,
                                  self.dtypes_s, 2, wx.RA_SPECIFY_COLS)
        sizer.Add(rb, 0, wx.ALIGN_CENTRE | wx.ALL, 5)

        # asdict
        cb = self.asdict_ctrl = wx.CheckBox(self, -1, "As Dictionary")
        cb.SetHelpText("Instead of exporting the table, export {key:list} dict " + \
                       "with first row/coumn providing keys")
        sizer.Add(cb, 0, wx.GROW | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.Bind(wx.EVT_CHECKBOX, self.EvtCheckBox, cb)

        # asdict BOX
        sbox = wx.StaticBox(self, -1, "Dict Properties")
        bsizer = wx.StaticBoxSizer(sbox, wx.VERTICAL)
        rb = self.keyloc_ctrl = wx.RadioBox(self, -1, "Key Location",
                                    wx.DefaultPosition, wx.DefaultSize,
                                    ['Col 0', 'Row 0'], 2, wx.RA_SPECIFY_COLS)
        bsizer.Add(rb, 0, wx.ALIGN_CENTRE | wx.ALL, 5)
        rb = self.kdt_ctrl = wx.RadioBox(self, -1, "Key dtype",
                                  wx.DefaultPosition, wx.DefaultSize,
                                  self.dtypes_s, 2, wx.RA_SPECIFY_COLS)
        bsizer.Add(rb, 0, wx.ALIGN_CENTRE | wx.ALL, 5)
        sizer.Add(bsizer, 0, wx.GROW | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)  # 1, wx.EXPAND|wx.ALL, 25)


        # name
        if default_name:
            hlp = "The name in global namespace which the table will " + \
                  "be assigned"
            box = wx.BoxSizer(wx.HORIZONTAL)

            label = wx.StaticText(self, -1, "Name:")
            label.SetHelpText(hlp)
            box.Add(label, 0, wx.ALIGN_CENTRE | wx.ALL, 5)

            text = self.name_ctrl = wx.TextCtrl(self, -1, default_name, size=(120, -1))
            text.SetHelpText(hlp)
            box.Add(text, 1, wx.ALIGN_CENTRE | wx.ALL, 5)

            sizer.Add(box, 0, wx.GROW | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        else:
            self.name_ctrl = False


        # bottom
        line = wx.StaticLine(self, -1, size=(20, -1), style=wx.LI_HORIZONTAL)
        sizer.Add(line, 0, wx.GROW | wx.ALIGN_CENTER_VERTICAL | wx.RIGHT | wx.TOP, 5)

        btnsizer = wx.StdDialogButtonSizer()

#        if wx.Platform != "__WXMSW__":
#            btn = wx.ContextHelpButton(self)
#            btnsizer.AddButton(btn)

        btn = wx.Button(self, wx.ID_OK)
        btn.SetHelpText("Export the table to the shell")
        btn.SetDefault()
        btnsizer.AddButton(btn)

        btn = wx.Button(self, wx.ID_CANCEL)
        btn.SetHelpText("Return to the table view without exporting the table")
        btnsizer.AddButton(btn)
        btnsizer.Realize()

        sizer.Add(btnsizer, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        self.SetSizer(sizer)
        sizer.Fit(self)

    def EvtCheckBox(self, event):
        pass
    def GetName(self):
        if self.name_ctrl:
            name = self.name_ctrl.GetValue()
            return name
        else:
            return False
    def GetKwargs(self):
        kwargs = {'dt': self.dtypes[self.dt_ctrl.GetSelection()],
                  'kdt': self.dtypes[self.kdt_ctrl.GetSelection()], }
        if self.asdict_ctrl.GetValue():
            if self.keyloc_ctrl.GetSelection() == 0:
                kwargs['asdict'] = 'col'
            else:
                kwargs['asdict'] = 'row'
        else:
            kwargs['asdict'] = False
        return kwargs



class TableFrame(wx.Frame, wx.FileDropTarget):
    def __init__(self, parent, table, rows=3, cols=3, title="Table", **kwargs):
        wx.Frame.__init__(self, parent, -1, title=title, **kwargs)
        wx.FileDropTarget.__init__(self)
        self.parent_shell = parent
        # toolbar
        tb = self.CreateToolBar(wx.TB_HORIZONTAL)
        tb.SetToolBitmapSize(size=(32, 32))

        tb.AddLabelTool(ID.PYDOC_LOAD, "Load", Icon("tango/actions/document-open"))
        self.Bind(wx.EVT_TOOL, self.OnFileOpen, id=ID.PYDOC_LOAD)

        # tb.AddLabelTool(ID.RESIZE, "resize", Icon("mine.table"))
        # self.Bind(wx.EVT_TOOL, self.OnTableResize, id=ID.RESIZE)
        tb.AddLabelTool(ID.TABLE_SAVE, "Save", Icon("tango/actions/document-save"))
        self.Bind(wx.EVT_TOOL, self.OnTableSave, id=ID.TABLE_SAVE)

        tb.AddSeparator()
        tb.AddLabelTool(ID.ATTACH, "attach", Icon("actions/attach"))
        self.Bind(wx.EVT_TOOL, self.OnAttach, id=ID.ATTACH)

        # tb.AddSeparator()
        txt = wx.StaticText(tb, -1, "Columns:")
        sc = wx.SpinCtrl(tb, ID.TABLE_COL, "3", min=0, max=1000, initial=cols)
        tb.AddControl(txt)
        tb.AddControl(sc)
        self.sc_c = sc
        txt = wx.StaticText(tb, -1, "Rows:")
        sc = wx.SpinCtrl(tb, ID.TABLE_ROW, "3", min=0, max=1000, initial=rows)
        tb.AddControl(txt)
        tb.AddControl(sc)
        self.sc_r = sc
        self.Bind(wx.EVT_SPINCTRL, self.OnSpinCtrl, id=-1)

        # tb.AddControl()

        tb.Realize()

        # grid
        g = MyGrid_simple(self)
        g.CreateGrid(rows, cols)
        self.grid = g
        if table:
            g.set(table)

        self.Sizer = wx.BoxSizer()
        self.Sizer.Add(g, 1, wx.EXPAND)

        self.Show()

    def OnSpinCtrl(self, event=None):
        id = event.GetId()
        n = event.GetInt()
        if id == ID.TABLE_ROW:
            n_now = self.grid.GetNumberRows()
            if n > n_now:
                self.grid.AppendRows(n - n_now)
            else:
                self.grid.DeleteRows(n, n_now - n)
        elif id == ID.TABLE_COL:
            n_now = self.grid.GetNumberCols()
            if n > n_now:
                self.grid.AppendCols(n - n_now)
            else:
                self.grid.DeleteCols(n, n_now - n)
    def SetColsRows(self, cols, rows):
        if self.grid.GetNumberCols() > 0:
            self.grid.DeleteCols(0, -1)
        if self.grid.GetNumberRows() > 0:
            self.grid.DeleteRows(0, -1)
        self.grid.AppendCols(cols)
        self.grid.AppendRows(rows)
        self.sc_c.SetValue(cols)
        self.sc_r.SetValue(rows)
    def AppendRow(self):
        self.grid.AppendRows(1)
        n = self.grid.GetNumberRows()
        self.sc_r.SetValue(n)
    def OnDropFiles(self, x, y, filenames):
        logging.debug("OnDropFiles: %s" % str(filenames))
        if type(filenames) == list:
            filename = filenames[0]
        else:
            filename = filenames
        self.load(filename)
    def OnTableResize(self, event=None):
        # dialog = wx.Dialog(self)
        logging.debug("OnTableResize")
        cols, rows = (5, 5)
        self.grid.resize(cols, rows)
    def load(self, path):
        table = loadtable(path)
        self.grid.set(table)
        self.SetTitle(path)
    def OnFileOpen(self, event=None):
        dialog = wx.FileDialog(self, "Select Table File",
                               style=wx.FD_OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            path = dialog.GetPath()
            self.load(path)
    def OnTableSave(self, event=None):
        dialog = wx.FileDialog(self, "Save Table",
                               wildcard="Pickle (*.pickled)|*.pickled|" + \
                                        "Tab Separated Values (*.txt)|*.txt",
                               style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        path = None
        while path == None:
            if dialog.ShowModal() == wx.ID_OK:
                path = dialog.GetPath()
                file_type = dialog.GetFilterIndex()
                path = add_ext(path, ['pickled', 'txt'][file_type])
            else:
                return

        if file_type == 0:  # pickle
            dialog = TableAttachDialog(self, False, title="Pickle Table",
                                       message="Settings for pickling table:")
            dialog.CenterOnScreen()
            if dialog.ShowModal() == wx.ID_OK:
                table = self.grid.get(**dialog.GetKwargs())
                with open(path, 'wb') as fid:
                    pickle.dump(table, fid)
        elif file_type == 1:  # TSV
            with open(path, 'w') as f:
                for row in self.grid.get():
                    f.write('\t'.join(row) + '\n')
    def OnAttach(self, event=None, name=None):
        g = self.parent_shell.global_namespace
        if not name:
            default_name = 't'; i = 1
            while default_name in g:
                default_name = 't%s' % i; i += 1
            dialog = TableAttachDialog(self, default_name)
            dialog.CenterOnScreen()
#            dialog = wx.TextEntryDialog(self, "Name for Table", "Name",
#                                        default_name)
            repeat = True
            while repeat:
                i = dialog.ShowModal()
                if i == wx.ID_OK:
                    repeat = False
                    name = dialog.GetName()
                    if name in g:
                        dialog2 = wx.MessageDialog(self, "Name '%s' is already present in global namespace. Overwrite?" % name,
                                                   "uga", wx.YES_NO | wx.NO_DEFAULT)
                        overwrite = dialog2.ShowModal()
                        if not overwrite:
                            repeat = True
                else:
                    return
            self.parent_shell.shell_message('%s = <table>' % name,
                                            internal_call=True)
            g[name] = self.grid.get(**dialog.GetKwargs())
            self.Close()

