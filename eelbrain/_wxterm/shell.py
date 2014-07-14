"""Modified PyShell class"""

from datetime import datetime
import inspect
import logging
import os
import cPickle as pickle
import re
import string
import sys
import tempfile
import types
import webbrowser

import wx
import wx.stc
import wx.lib.colourselect
import wx.py.shell
from wx.py.frame import ID_AUTO_SAVESETTINGS, ID_COPY_PLUS, ID_SAVEHISTORY

import matplotlib as mpl
import matplotlib.pyplot as plt

from .. import __version__
from .. import fmtxt
from .._utils import ui
from .._utils import print_funcs
from .._wxutils import droptarget, Icon, ID
from .._wxutils.mpl_canvas import CanvasFrame
from .about_dialog import AboutFrame
from .help import HelpFrame
from .mpl_tools import PyplotManager
from .preferences_dialog import PreferencesDialog
from .py_editor import PyEditor
from .table import TableFrame


history_session_hdr = "#  Eelbrain Session:  %s"
_punctuation = string.punctuation.replace('.', '').replace('_', '')


def is_py_char(char):
    return (char.isalnum() or char == '_')


def is_py_varname(name):
    a = re.match('^[a-zA-Z_]', name)
    b = re.match('[a-zA-Z0-9_]', name)
    return a and b


# modify wx.py introspection to take into account __wrapped__
from wx.py.introspect import getConstructor
def getBaseObject(obj):
    """Return base object and dropSelf indicator for an object."""
    if inspect.isbuiltin(obj):
        # Builtin functions don't have an argspec that we can get.
        dropSelf = 0
    elif inspect.ismethod(obj):
        # Get the function from the object otherwise
        # inspect.getargspec() complains that the object isn't a
        # Python function.
        try:
            if obj.im_self is None:
                # This is an unbound method so we do not drop self
                # from the argspec, since an instance must be passed
                # as the first arg.
                dropSelf = 0
            else:
                dropSelf = 1
            obj = obj.im_func
        except AttributeError:
            dropSelf = 0
    elif inspect.isclass(obj):
        # Get the __init__ method function for the class.
        constructor = getConstructor(obj)
        if constructor is not None:
            obj = constructor
            dropSelf = 1
        else:
            dropSelf = 0
    elif callable(obj):
        # Get the __call__ method instead.
        try:
            obj = obj.__call__.im_func
            dropSelf = 1
        except AttributeError:
            dropSelf = 0
    else:
        dropSelf = 0

    #  MY MOD
    obj = getattr(obj, '__wrapped__', obj)
    # END MY MOD

    return obj, dropSelf
wx.py.introspect.getBaseObject = getBaseObject


# subclass Shell in order to set some custom properties
class Shell(wx.py.shell.Shell):
    def __init__(self, *args, **kwargs):  # Leads to recursion crash
        self.exec_mode = 0  # counter to determine whether other objects than
        # the shell itself are writing to writeOut
        self.has_moved = False  # keeps track whether any entity has written
        # to the shell in exec_mode
        super(Shell, self).__init__(*args, **kwargs)

    def writeOut(self, message):
        """

        sep = False: adds linebreaks at the top and at the bottom of message

        ascommand = False: adds the prompt in front of the message to mimmick
                   command
        """
        if self.exec_mode:
            message = unicode(message)
            message = self.fixLineEndings(message)

            start = self.promptPosStart  # start des prompt >>>
            end = self.promptPosEnd  # end   des prompt >>>
#            current = self.CurrentPos    # caret position
#            last = self.GetLastPosition()# end of the text
#            logging.debug("WRITEOUT POS: start %s; end %s; current %s; last %s"%(start, end, current, last))

#            self.SetCurrentPos(end)#start)
#            self.SetAnchor(start)
            self.SetSelection(start, end)
            self.ReplaceSelection(message)
            self.has_moved = True
#            self.AddText(message)

            new_start = self.GetCurrentPos()
#            new_end = new_start + (end-start)
            self.promptPosStart = new_start
            self.promptPosEnd = new_start  # new_end
#
#            self.SetCurrentPos(new_end)
#            self.SetAnchor(new_end)
        else:
            # same as done by:  wx.py.shell.Shell.writeOut(self, message)
            self.write(message)

    def writeErr(self, message):
        # TODO: color=(0.93359375, 0.85546875, 0.3515625)
#        ls = os.linesep
#        if message != ls:
#            message = message.replace(ls, ls+'!   ')
        message = message.replace('"""', '"')  # some deprecation errors contain """ messing up shell colors
        self.writeOut(message)

    def start_exec(self):
        self.exec_mode += 1

    def end_exec(self):
        self.exec_mode -= 1
        if self.exec_mode == 0 and self.has_moved:
            self.prompt()
            self.has_moved = False

    def showIntro(self, text=''):
        """Display introductory text in the shell."""
        if text:
            self.write(text)
        try:
            if self.interp.introText:
                if text and not text.endswith(os.linesep):
                    self.write(os.linesep)
                # modified to avoid printing invalid instructions
                for line in self.interp.introText.splitlines():
                    if line.startswith('Type "help"'):
                        pass
                    else:
                        self.write(line)
                # end modification
        except AttributeError:
            pass


class ShellFrame(wx.py.shell.ShellFrame):
    bufferOpen = 1  # Dummy attr so that py.frame enables Open menu
    bufferNew = 1  # same for New menu command
    bufferClose = 1  # same for Close menu command (handled by OnFileClose)
    def __init__(self, parent=None, app=None, title='Eelbrain Shell'):

    # --- set up PREFERENCES ---
        self.config = config = wx.Config("Eelbrain")
        std_paths = wx.StandardPaths.Get()

        # redirect stdio for debugging
        if app is not None:
            redirect = config.ReadBool('Debug/Redirect', False)
            if redirect:
                filename = config.Read('Debug/Logfile') or None
                app.RedirectStdio(filename)

        # override pyshell defaults
        if not config.HasEntry('Options/AutoSaveSettings'):
            config.WriteBool('Options/AutoSaveSettings', True)
        if not config.HasEntry('Options/AutoSaveHistory'):
            config.WriteBool('Options/AutoSaveHistory', True)

        # get pyshell dataDir
        config_dir = os.path.join(std_paths.GetUserConfigDir(), 'Eelbrain')
        if not os.path.exists(config_dir):
            os.mkdir(config_dir)

        # make sure startup file exists
        startup_file = os.path.join(config_dir, 'startup')
        if not os.path.exists(startup_file):
            with open(startup_file, 'w') as fid:
                fid.write("# Eelbrain startup script\n")

        # http://wiki.wxpython.org/FileHistory
        self.filehistory = wx.FileHistory(10)
        self.filehistory.Load(config)

    # --- SHELL initialization ---
        # put my Shell subclass into wx.py.shell
        wx.py.shell.Shell = Shell
        wx.py.shell.ShellFrame.__init__(self, parent=parent, title=title,
                                        config=config, dataDir=config_dir)

        self.SetStatusText('Eelbrain %s' % __version__)

        droptarget.set_for_strings(self.shell)

        # attr
        self.global_namespace = self.shell.interp.locals
        self.editors = []
        self.active_editor = None  # editor last used; updated by Editor.OnActivate and Editor.__init__
        self.tables = []
        self.experiments = []  # keep track of ExperimentFrames
        self._attached_items = {}
        self.help_viewer = None

        # load history/configuration
        if len(self.shell.history):
            self.shell.history.pop(-1)
        self.LoadSettings()
        now = datetime.now()
        info = history_session_hdr % now.isoformat(' ')[:16]
        self.shell.history.insert(0, info)

    # child windows
        x_min, y_min, x_max, y_max = wx.Display().GetGeometry()
        self.P_mgr = PyplotManager(self, pos=(x_max - 100, y_min + 22 + 4))

    # --- MENUS ---
        # wx 2.8 somehow does not manage to access the 'window' and 'help'
        # menus; this works in 2.9

        def get_menu_Id(name):
            for Id in xrange(self.menuBar.GetMenuCount()):
                if self.menuBar.GetMenuLabel(Id) == name:
                    return Id

        # for debugging:
        # create a name->Id dict for the current menus
        # (created by wx.py.frame.Frame.__createMenus() )
#        self.menu_names = {self.menuBar.GetMenuLabel(i): i for i in
#                           xrange(self.menuBar.GetMenuCount())}

    # recent menu
        recent_menu = self.recent_menu = wx.Menu()
        self.filehistory.UseMenu(recent_menu)
        self.filehistory.AddFilesToMenu()
        self.recent_menu_update_icons()

        self.Bind(wx.EVT_MENU_RANGE, self.OnRecentItemLoad, id=wx.ID_FILE1, id2=wx.ID_FILE9)

        # add menu item (can only add it to one menu apparently)
        help_txt = "Load an experiment or Python script from a list of recently used items"
        self.fileMenu.InsertMenu(0, wx.ID_ANY, 'Recent Files', recent_menu, help_txt)

    # preferences menu
        if wx.Platform == '__WXMAC__':
            app = wx.GetApp()
            ID_PREFERENCES = app.GetMacPreferencesMenuItemId()
        else:  # although these seem to be the same -- 5022
            ID_PREFERENCES = wx.ID_PREFERENCES
        self.optionsMenu.Append(ID_PREFERENCES, "&Preferences...\tCtrl+,")
        self.Bind(wx.EVT_MENU, self.OnPreferences, id=wx.ID_PREFERENCES)

    # add "Open Examples" to file menu
        ex_root = __file__
        for _ in xrange(3):
            ex_root, _ = os.path.split(ex_root)
        self._examples_root = os.path.join(ex_root, 'examples')
        self.fileMenu.Insert(3, ID.SHOW_EXAMPLES, "Open Example...")
        self.Bind(wx.EVT_MENU, self.OnShowExamples, id=ID.SHOW_EXAMPLES)

    # add "Open History" to file menu
        m = self.historyFileMenu = wx.Menu()

        # collect history by date
        self._history_items = {}
        i_stop = 0
        header = history_session_hdr % ""
        header_len = len(header)
        for i, line in enumerate(self.shell.history[1:], 1):
            if line.startswith(header):
                i_start = i - 1
                n_items = i_start - i_stop
                if n_items:
                    history = self.shell.history[i_start:i_stop:-1]
                    txt = os.linesep.join(history)
                    n_lines = txt.count(os.linesep) + 1
                    name = line[header_len:]
                    label = name + ' (%i lines)' % n_lines
                    id_ = wx.NewId()
                    m.Append(id_, label, "Open history from %s." % label)
                    self.Bind(wx.EVT_MENU, self.OnOpenHistory, id=id_)
                    self._history_items[id_] = (name, txt)
                i_stop = i

        m.PrependSeparator()
        m.Prepend(ID.OPEN_HISTORY_CURRENT, "Current Session", "Open the "
                  "history for the current session in a Python document.")
        m.Prepend(ID.OPEN_HISTORY, "Entire History", "Open the entire history "
                  "as a new Python document.")
        self.fileMenu.InsertMenu(1, ID.OPEN_HISTORY, "History", m, "Open the "
                                 "command history in a Python script editor.")
        self.Bind(wx.EVT_MENU, self.OnOpenHistory, id=ID.OPEN_HISTORY_CURRENT)
        self.Bind(wx.EVT_MENU, self.OnOpenHistory, id=ID.OPEN_HISTORY)

    # edit menu
        if wx.Platform == '__WXMAC__':
            id_ = self.editMenu.FindItem('Redo \t')
            item = self.editMenu.FindItemById(id_)
            item.SetItemLabel('Redo \tctrl-shift-z')

    # Exec Menu
        m = self.execMenu = wx.Menu()
        m.Append(ID.EXEC_SELECTION, "Selection or Line  \tCtrl+Enter",
                 "Execute the currently selected code or the current line.")
        m.Append(ID.EXEC_DOCUMENT, "Document", "Execute the front most Python "
                 "document.")
        m.Append(ID.EXEC_DOCUMENT_FROM_DISK, "Document from Disk  \tCtrl+E",
                 "Save and execute the Python document from disk.")
        self.menuBar.Insert(get_menu_Id('&Help'), m, "Exec")

        self.Bind(wx.EVT_MENU, self.OnExecSelection, id=ID.EXEC_SELECTION)
        self.Bind(wx.EVT_MENU, self.OnExecDocument, id=ID.EXEC_DOCUMENT)
        self.Bind(wx.EVT_MENU, self.OnExecDocumentFromDisk,
                  id=ID.EXEC_DOCUMENT_FROM_DISK)

        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateMenu, id=ID.EXEC_SELECTION)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateMenu, id=ID.EXEC_DOCUMENT)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateMenu,
                  id=ID.EXEC_DOCUMENT_FROM_DISK)

    # MNE Menu
        # (without requiring mne-python to be installed)
        try:
            import mne
            m = self.mneMenu = wx.Menu()

            self.mne_cmds = {}
            for cmd in sorted(dir(mne.gui)):
                if cmd.startswith('_'):
                    continue

                # check that the command has no obligatory parameters
                func = getattr(mne.gui, cmd)
                argspec = inspect.getargspec(func)
                args = argspec.args
                if args:
                    defaults = argspec.defaults
                    if (defaults is None) or (len(args) > len(defaults)):
                        continue

                # add menu item
                Id = wx.NewId()
                self.mne_cmds[Id] = cmd
                name = cmd.capitalize()
                m.Append(Id, name, "Open the mne %s GUI" % name)
                self.Bind(wx.EVT_MENU, self.OnOpenMneGui, id=Id)

            self.menuBar.Insert(get_menu_Id("&Help"), m, "MNE")
        except:
            pass

    # WINDOW MENU
        Id = get_menu_Id("&Window")
        if Id is None:
            m = self.windowMenu = wx.Menu()  # title="&Window")
            m.SetTitle("&Window")
            Id_help = get_menu_Id('&Help')
            self.menuBar.Insert(Id_help, m, 'Window')
        else:
            m = self.windowMenu

        # name is used below in OnOpenWindowMenu to recognize Window menu
        self.Bind(wx.EVT_MENU_OPEN, self.OnOpenWindowMenu)

        m.Append(ID.PYPLOT_DRAW, '&Pyplot Draw  \tCtrl+P', 'Draw all figures.')
        self.Bind(wx.EVT_MENU, self.OnPyplotDraw, id=ID.PYPLOT_DRAW)

        m.Append(ID.P_MGR, "Pyplot Manager",
                 "Toggle Pyplot Manager Panel.")  # , wx.ITEM_CHECK)
        self.Bind(wx.EVT_MENU, self.OnTogglePyplotMgr, id=ID.P_MGR)

        m.Append(ID.PYPLOT_CLOSEALL, "Close All Plots",
                 "Closes all Matplotlib plots.")
        self.Bind(wx.EVT_MENU, self.OnCloseAllPlots, id=ID.PYPLOT_CLOSEALL)

        # shell resizing
        m.AppendSeparator()
        m.Append(ID.FOCUS_SHELL, "Focus Shell Prompt \tCtrl+L", "Raise the "
                 "shell and bring the caret to the end of the prompt")
        self.Bind(wx.EVT_MENU, self.OnFocusPrompt, id=ID.FOCUS_SHELL)

        m.Append(ID.SIZE_MAX, 'Maximize Shell',
                 "Expand the shell to use the full screen height.")
        self.Bind(wx.EVT_MENU, self.OnResize, id=ID.SIZE_MAX)

        m.Append(ID.SIZE_MAX_NOTITLE, 'Maximize Hide Title',
                 "Expand the shell and hide the title bar.")
        self.Bind(wx.EVT_MENU, self.OnResize, id=ID.SIZE_MAX_NOTITLE)

        m.Append(ID.SIZE_MIN, 'Mini-Shell', "Resize the Shell to a small "
                 "window (e.g. for use as pocket calculator).")
        self.Bind(wx.EVT_MENU, self.OnResize, id=ID.SIZE_MIN)

        # section with all open windows
        m.AppendSeparator()
        item = m.Append(self.GetId(), "Shell", "Bring shell to the front.")
        self.Bind(wx.EVT_MENU, self.OnWindowMenuActivateWindow, item)
        self.windowMenuWindows = {self.GetId(): self}
        self.windowMenuMenuItems = {}

        # clear shell: this can be done with edit->empty buffer

    # EDIT MENU
        m = self.editMenu
        m.AppendSeparator()
        m.Append(ID.COMMENT, "Comment Line \tCtrl+/", "Comment in/out current "
                 "line")
        self.Bind(wx.EVT_MENU, self.OnComment, id=ID.COMMENT)
        m.Append(ID.DUPLICATE, "Duplicate Commands \tCtrl-D")
        self.Bind(wx.EVT_MENU, self.OnDuplicate, id=ID.DUPLICATE)
        m.Append(ID.DUPLICATE_WITH_OUTPUT, "Duplicate Plus \tCtrl-Shift-D")
        self.Bind(wx.EVT_MENU, self.OnDuplicate, id=ID.DUPLICATE_WITH_OUTPUT)

        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateMenu, id=ID.COMMENT)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateMenu, id=ID.DUPLICATE)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateMenu, id=ID.DUPLICATE_WITH_OUTPUT)


    # INSERT MENU
        m = self.insertMenu = wx.Menu()
        m.Append(ID.INSERT_Color, "Color", "Insert color as (r, g, b)-tuple.")
#        self.Bind(wx.EVT_MENU, self.OnInsertColor, id=ID.INSERT_Color)
#        this function only works for the colorpicker linked to the button

        # path submenu
        m = self.pathMenu = wx.Menu()
        m.Append(ID.INSERT_Path_file, "File",
                 "Insert path to an existing File (You can also simply "
                 "drag a file on the shell or an editor).")
        m.Append(ID.INSERT_Path_dir, "Directory",
                 "Insert path to an existing Directory (You can also "
                 "simply drag a file on the shell or an editor).")
        m.Append(ID.INSERT_Path_new, "New",
                 "Insert path to a non-existing object.")
        self.insertMenu.AppendSubMenu(m, "Path", "Insert a path as string.")

        Id_view = get_menu_Id('&View')
        self.menuBar.Insert(Id_view, self.insertMenu, "Insert")
        self.Bind(wx.EVT_MENU, self.OnInsertPath_File, id=ID.INSERT_Path_file)
        self.Bind(wx.EVT_MENU, self.OnInsertPath_Dir, id=ID.INSERT_Path_dir)
        self.Bind(wx.EVT_MENU, self.OnInsertPath_New, id=ID.INSERT_Path_new)

    # OPTIONS menu
        # remove the conflicting keyboard shortcut
        m = self.optionsMenu.FindItemByPosition(0)
        sm = m.GetSubMenu()
        i = sm.FindItemByPosition(3)
        i.SetItemLabel(i.ItemLabel[:-13])

    # HELP menu
        if wx.__version__ >= '2.9':
            m = self.helpMenu
            m.AppendSeparator()
        else:
            m = wx.Menu()
            Id = self.menuBar.GetMenuCount()
            self.menuBar.Insert(Id, m, "Online Help")


        m.Append(ID.HELP_EELBRAIN, "Eelbrain", "Open the Eelbrain "
                 "documentation pages in an external browser.")
        self.Bind(wx.EVT_MENU, self.OnHelpExternal, id=ID.HELP_EELBRAIN)

        m.Append(ID.HELP_PYTHON, "Python", "Open the official Python "
                 "documentation page in an external browser.")
        self.Bind(wx.EVT_MENU, self.OnHelpExternal, id=ID.HELP_PYTHON)

        m.Append(ID.HELP_PDB, "   pdb (Python Debugger)", "Open the Python "
                 "Debugger documentation page in an external browser.")
        self.Bind(wx.EVT_MENU, self.OnHelpExternal, id=ID.HELP_PDB)

        m.Append(ID.HELP_MPL, "Matplotlib (Pyplot)", "Open the Matplotlib "
                 "Pyplot reference page in an external browser.")
        self.Bind(wx.EVT_MENU, self.OnHelpExternal, id=ID.HELP_MPL)

        m.Append(ID.HELP_MDP, "mdp",
                 "Open the mdp Documentation page in an external browser.")
        self.Bind(wx.EVT_MENU, self.OnHelpExternal, id=ID.HELP_MDP)


    # --- TOOLBAR ---
        tb = self.CreateToolBar(wx.TB_HORIZONTAL)
        tb.SetToolBitmapSize(size=wx.Size(32, 32))

        # windows/tools
        tb.AddLabelTool(wx.ID_HELP, "Help", Icon("tango/apps/help-browser"),
                        shortHelp="Open the Help-Viewer",
                        longHelp="Open the Help-Viewer")
        self.Bind(wx.EVT_TOOL, self.OnHelpViewer, id=wx.ID_HELP)

        tb.AddLabelTool(ID.P_MGR, "Pyplot Manager",
                        Icon("tango/mimetypes/x-office-presentation"),
                        shortHelp="Toggle display of the Pyplot manager tool")
        self.Bind(wx.EVT_TOOL, self.OnTogglePyplotMgr, id=ID.P_MGR)
        tb.AddSeparator()

        # File Operations
        tb.AddLabelTool(wx.ID_OPEN, "Load",
                        Icon("tango/actions/document-open"), shortHelp="Load "
                        "a Python document (*.py)")

        tb.AddLabelTool(ID.EXEC_FILE, "Exec", Icon("documents/pydoc-openrun"),
                        shortHelp="Run an existing Python script (without "
                        "opening it in an editor)")
        self.Bind(wx.EVT_TOOL, self.OnExecFile, id=ID.EXEC_FILE)

        tb.AddLabelTool(wx.ID_NEW, "New .Py Document", Icon("documents/pydoc-new"),
                        shortHelp="Open a new Python script editor")

        tb.AddLabelTool(ID.TABLE, "Table", Icon("documents/table"),
                        shortHelp="Open a new table")
        self.Bind(wx.EVT_TOOL, self.OnTableNew, id=ID.TABLE)
        tb.AddSeparator()

        # inserting (these controls have no shortHelp)
        fn_ctrl = wx.Choice(tb, wx.ID_FIND, (100, 50), choices=['File', 'Dir', 'New'])
        self.Bind(wx.EVT_CHOICE, self.OnInsertPath, id=wx.ID_FIND)
        tb.AddControl(fn_ctrl)
        b = wx.lib.colourselect.ColourSelect(tb, -1, "Color", (0, 255, 0),
                                             size=wx.DefaultSize)
        b.Bind(wx.lib.colourselect.EVT_COLOURSELECT, self.OnInsertColor)
        self.color_selector = b
        # bind the 'insert color' menu entry
        self.Bind(wx.EVT_MENU, self.color_selector.Command, id=ID.INSERT_Color)
        tb.AddControl(b)
        # tb.AddLabelTool(ID.COLOUR_CHOOSER, "Color", Icon('apps.preferences-desktop-locale'))
        # self.Bind(wx.EVT_TOOL, self.OnSelectColourAlt, id=ID.COLOUR_CHOOSER)

        if wx.__version__ >= '2.9':
            tb.AddStretchableSpace()
        else:
            tb.AddSeparator()

        # TODO: clear only the last command + output
        tb.AddLabelTool(ID.CLEAR_TERMINAL, "Clear Text", Icon("tango/actions/edit-clear"),
                        shortHelp="Clear all text from the terminal")
        self.Bind(wx.EVT_TOOL, self.OnClearTerminal, id=ID.CLEAR_TERMINAL)

        tb.AddLabelTool(wx.ID_EXIT, "Quit", Icon("tango/actions/system-log-out"),
                        shortHelp="Quit eelbrain")
        self.Bind(wx.EVT_TOOL, self.OnQuit, id=wx.ID_EXIT)

        tb.Realize()


    # --- Finalize ---

        # add commands to the shell
        self.global_namespace['attach'] = self.attach
        self.global_namespace['detach'] = self.detach
        self.global_namespace['help'] = self.help_lookup
        if wx.__version__ < '2.9':
            self.global_namespace['cd'] = self.curdir

        for name in ('printdict', 'printlist', 'dicttree'):
            self.global_namespace[name] = getattr(print_funcs, name)

        # other Bindings
        self.Bind(wx.EVT_MAXIMIZE, self.OnMaximize)
        self.Bind(wx.EVT_ACTIVATE, self.OnActivate)
        # my shell customization
        self.ApplyStyle()
        self.Resize(ID.SIZE_MAX)
        # icon
        if sys.platform != 'darwin':
            self.eelbrain_icon = Icon('eelbrain', asicon=True)
            self.SetIcon(self.eelbrain_icon)
            self.Bind(wx.EVT_CLOSE, self.OnDestroyIcon)


        # add help text from wx.py.shell
        text = wx.py.shell.HELP_TEXT
        self.__doc__ = text

        self.preferencesDialog = None

    def ApplyStyle(self):
        "reapply the layout to all editwindows"
#        if 'wxMac' in wx.PlatformInfo:
        self._style = {'times'     : 'Lucida Grande',
                       'mono'      : self.config.Read('font', 'Monaco'),  # 'Courier New'
                       'helv'      : 'Geneva',
                       'other'     : 'new century schoolbook',
                       'size'      : int(self.config.Read('font size', '13')),
                       'lnsize'    : 16,
                       'forecol'   : self.config.Read('font color', '#FFFF00'),
                       'backcol'   : '#F0F0F0',  # '#000010',
                       'calltipbg' : '#101010',
                       'calltipfg' : '#BBDDFF',
                       }

        self.ApplyStyleTo(self.shell)
        for ed in self.editors:
            if hasattr(ed, 'editor'):  # check if alife
                self.ApplyStyleTo(ed.editor.window)

    def ApplyStyleTo(self, obj):
        """
        see
         - wx.py.editwindow.EditWindow.setStyles
         - wx.py.shell.editwindow.EditWindow.__config()

        """
        obj.SetCaretWidth(2)

        obj.StyleClearAll()
        FACES = self._style

        obj.setStyles(FACES)  # this calls wx.py.editwindow.EditWindow.setStyles
        obj.CallTipSetBackground(FACES['calltipbg'])
        obj.CallTipSetForeground(FACES['calltipfg'])

        obj.StyleSetSpec(wx.stc.STC_STYLE_DEFAULT, "fore:%(forecol)s" % FACES)  # ,back:#666666")
        obj.StyleSetSpec(wx.stc.STC_P_DEFAULT, "fore:#000000,face:%(mono)s" % FACES)  # '\' space
        obj.StyleSetSpec(wx.stc.STC_P_NUMBER, "fore:#0000FF")
        obj.StyleSetSpec(wx.stc.STC_STYLE_CONTROLCHAR, "fore:#FF0000")
        obj.StyleSetSpec(wx.stc.STC_P_STRING, "fore:#FF0000")
        obj.StyleSetSpec(wx.stc.STC_P_CHARACTER, "fore:#0000FF")
        obj.StyleSetSpec(wx.stc.STC_P_OPERATOR, "fore:#000010,face:%(mono)s" % FACES)
        obj.StyleSetSpec(wx.stc.STC_P_DECORATOR, "fore:#FF00FF")
        obj.StyleSetSpec(wx.stc.STC_P_WORD, "fore:#FF00FF")
        obj.StyleSetSpec(wx.stc.STC_STYLE_MAX, "fore:#0000FF")
        comment = "fore:#008811"
        obj.StyleSetSpec(wx.stc.STC_P_COMMENTLINE, comment)
        obj.StyleSetSpec(wx.stc.STC_P_TRIPLEDOUBLE, "%s,back:#F0F0F0" % comment)
        obj.StyleSetSpec(wx.stc.STC_P_TRIPLE, comment)

    def attach(self, dictionary, detach=True, _internal_call=False):
        """
        Adds a dictionary to the globals and keeps track of the items so that
        they can be removed safely with the detach function. Also works for
        modules (in which case any private attributes are ignored).

        detach : bool
            Detach anything else before attaching the current dictionary.

        """
        if detach:
            self.detach()

        # detect and convert special types
        if isinstance(dictionary, types.ModuleType):
            mod = vars(dictionary)
            dictionary = {k:mod[k] for k in mod if not k.startswith('_')}

        # self._attached_items = {id(dict_like) -> {key -> value}}
        present = []
        for k in dictionary:
            if not isinstance(k, basestring):
                raise ValueError("Dictionary contains non-strong key: %r" % k)
            elif not is_py_varname(k):
                raise ValueError("Dictionary contains invalid name: %r" % k)
            elif k in self.global_namespace:
                if id(self.global_namespace[k]) != id(dictionary[k]):
                    present.append(k)

        if present:
            title = "Overwrite Items?"
            message = ("The following items are associated with different "
                       "objects in global namespace. Should we replace them?"
                       + os.linesep)
            message = os.linesep.join((message, ', '.join(repr(k) for k in present)))
            answer = ui.ask(title, message, True)
            if answer is None:
                return
            elif answer:
                attach = dictionary
            else:
                attach = dict((k, v) for k, v in dictionary.iteritems() if k not in present)
        else:
            attach = dictionary

        self.global_namespace.update(attach)

        items = self._attached_items.setdefault(id(dictionary), {})
        items.update(attach)

        msg = "attached: %s" % str(attach.keys())
        self.shell_message(msg, internal_call=_internal_call)

    def bufferSave(self):
        "to catch an distribute menu command 'save' in Os-X"
        if self.IsActive():
            wx.py.shell.ShellFrame.bufferSave(self)
        else:
            for e in self.editors:
                if e.IsActive():
                    e.bufferSave()

    def bufferSaveAs(self):
        "to catch an distribute menu command 'save as' in Os-X"
        if self.IsActive():
            wx.py.shell.ShellFrame.bufferSave(self)
        else:
            for e in self.editors:
                if e.IsActive():
                    e.bufferSaveAs()

    def CloseAllPlots(self):
        plt.close('all')
        for frame in self.Children:
            if isinstance(frame, CanvasFrame):
                frame.Close()

    def create_py_editor(self, pyfile=None, name="Script Editor"):
        """
        Creates and returns a new py_editor object.

        :arg pyfile: filename as string, or True in order to display an "open
            file" dialog (None creates an empty editor)
        :arg openfile: True if an 'open file' dialog should be shown after the
            editor is cerated

        """

        display = wx.Display()
        area = display.GetClientArea()

        shell = self.GetRect()

        w = 640
        h = area.height
        size = (w, h)

        x = min(shell.right, area.right - w)
        y = area.top
        pos = (x, y)

        editor = PyEditor(self, self, pos=pos, size=size, pyfile=pyfile,
                          name=name)

        editor.Show()
        self.editors.append(editor)

        # add to window menu
        ID = editor.GetId()
        m = self.windowMenu.Append(ID, "a", "Bring window to the front.")
        self.Bind(wx.EVT_MENU, self.OnWindowMenuActivateWindow, m)
        self.windowMenuWindows[ID] = editor
        self.windowMenuMenuItems[ID] = m

        return editor

    def curdir(self, dirname=None):
        """
        Set the current directory. With None, returns the current directory.

        """
        if dirname:
            dirname = os.path.expanduser(dirname)
            os.chdir(dirname)
            if dirname not in sys.path:
                if hasattr(self, '_added_to_path'):
                    sys.path.remove(self._added_to_path)
                sys.path.append(dirname)
                self._added_to_path = dirname
        else:
            return os.path.abspath(os.curdir)

    def detach(self, dictionary=None):
        """
        Removes the contents of `dictionary` from the global namespace. Neither
        the item itself is removed, nor are any references that were not
        created through the :func:`attach` function.

        `dictionary`: dict-like
            Dictionary which to detach; `None`: detach everything

        """
        if dictionary is None:
            dIDs = self._attached_items.keys()
        else:
            dIDs = [id(dictionary)]

        detach = {}
        for dID in dIDs:
            detach.update(self._attached_items.pop(dID))

        for k, v in detach.iteritems():
            if id(self.global_namespace[k]) == id(v):
                del self.global_namespace[k]

    def Duplicate(self, output=False):
        # make sure we have a target editor
        if not hasattr(self.active_editor, 'InsertLine'):
            self.OnFileNew()
            self.Raise()

        editor = self.active_editor

        # prepare text for transfer
        text = self.shell.GetSelectedText()
        lines = [line for line in text.split(os.linesep) if len(line) > 0]

        for line in lines:
            if output:
                editor.InsertLine(line)
            else:
                line_stripped = self.shell.lstripPrompt(line)
                if len(line_stripped) < len(line):
                    editor.InsertLine(line_stripped)

    def DuplicateFull(self):
        self.Duplicate(True)

    def ExecCommand(self, cmd):
        """Execute a single command in the shell. Can be multiline command."""
        # clean command
        cmd = fmtxt.unindent(cmd.rstrip())
        multiline = '\n' in cmd

        # insert cmd into shell and process
        self.shell.clearCommand()
        self.InsertStrToShell(cmd)
        self.shell.processLine()
        if multiline:
            self.shell.processLine()

    def ExecFile(self, filename=None, shell_globals=True):
        """
        Execute a file in the shell.

        Parameters
        ----------
        filename : None | str
            File to execute. If None, an open file dialog is used.
        shell_globals : bool
            Wheter the file should be executed in the shell's global namespace
            (or in a separate namespace).
        """
        if filename is None:
            dialog = wx.FileDialog(self, style=wx.FD_OPEN)
            dialog.SetMessage("Select Python File")
            dialog.SetWildcard("Python files (*.py)|*.py")
            if dialog.ShowModal() == wx.ID_OK:
                filename = dialog.GetPath()
            else:
                return

        if not isinstance(filename, basestring):
            err = ("Invalid filename type:  needs to be str or unicode, not "
                   "%s" % repr(filename))
            logging.error("shell.ExecFile - " + err)
            raise TypeError(err)

        if not os.path.exists(filename):
            msg = "Trying to execute non-existing file: %r" % filename
            wx.MessageBox(msg, "Error Executing File", wx.ICON_ERROR | wx.OK)
            return

        # set paths in environment
        dirname = os.path.dirname(filename)
        self.curdir(dirname)

        if shell_globals:
            self.global_namespace['__file__'] = filename
            self.shell.Execute("execfile(%r)" % filename)
        else:
            if float(sys.version[:3]) >= 2.7:
                if 'runpy' not in self.global_namespace:
                    self.shell.Execute("import runpy")
                self.shell.Execute("out_globals = runpy.run_path(%r)" % filename)
            else:
                command = "execfile(%r, dict(__file__=%r))"
                self.shell.Execute(command % (filename, filename))

        self.pyplot_draw()

    def ExecText(self, txt, out=False, title="unknown source", comment=None,
                 shell_globals=True, filepath=None, internal_call=False):
        """
        Compile txt and Execute it in the shell.

        Parameters
        ----------
        txt : str
            Code to execute.
        out : bool
            Use shell.Execute
        title:
            is displayed in the shell and should identify the source of the
            code.
        comment:
            displayed after title
        shell_globals:
            determines wheter the shell's globals are submitted to
            the call to execfile or not.
        filepath:
            Perform os.chdir and set __file__ before executing.
        """
        if comment is None:
            msg = '<exec %r>' % title
        else:
            msg = '<exec %r, %s>' % (title, comment)
        self.shell_message(msg, ascommand=False, internal_call=internal_call)

        if filepath:
            self.curdir(os.path.dirname(filepath))
            if out or shell_globals:
                self.global_namespace['__file__'] = filepath

        if out:
            self.shell.Execute(txt)
        else:
            self.shell.start_exec()

            # this does not work
#            self.shell.interp.runsource(txt)

            # prepare txt
            txt = self.shell.fixLineEndings(txt)
            txt = txt.replace("'''", '"""')
            txt += os.linesep

            # remove leading whitespaces (and comment lines)
            lines = [line for line in txt.splitlines() if not line.startswith('#')]
            ws_lead = []
            for line in lines:
                len_stripped = len(line.lstrip(' '))
                if len_stripped:
                    ws_lead.append(len(line) - len_stripped)
            rm = min(ws_lead)
            if rm:
                new_lines = []
                for line in lines:
                    new_lines.append(line[rm:] if len(line) > rm else line)
                txt = os.linesep.join(new_lines)

            # 1
            cmd = "exec(compile(r'''{txt}\n''', '{title}', 'exec'), {globals})"
            if shell_globals:
                exec_globals = "globals()"
            else:
                exec_globals = "{'__file__': %r}" % filepath

            code = cmd.format(txt=txt, title=os.path.split(title)[-1],
                              globals=exec_globals)
            self.shell.push(code, silent=True)

            # 2
#            cmp = compile(txt, os.path.split(title)[-1], 'exec')
#            self.shell.push(cmp, silent=True)

            # 3 (runs onliy first line)
#            self.shell.interp.runsource(txt)

            self.shell.end_exec()

#        self.Raise() # (causes Shell to raise above new plots)

        self.pyplot_draw()

    def FrameTable(self, table=None):
        pos = self.pos_for_new_window()
        t = TableFrame(self, table, pos=pos)
        self.tables.append(t)

    def get_active_window(self):
        "returns the active window (self, editor, help viewer, ...)"
        if self.IsActive():
            return self
        for c in self.Children:
            if hasattr(c, 'IsActive') and c.IsActive():
                return c
        for w in  wx.GetTopLevelWindows():
            if hasattr(w, 'IsActive') and w.IsActive():
                return w
        return wx.GetActiveWindow()

    def GetCurLine(self):
        return self.shell.GetCurLine()

    def hasBuffer(self):
        return True

    def help_lookup(self, what=None):
        """
        what = object whose docstring should be displayed

        """
        # this is the function that is pulled into the shell as help()
        self.OnHelpViewer(topic=what)

    def InsertStr(self, text):
        """
        Inserts the text in the active window (can be the shell or an editor)
        """
        for editor in self.editors:
            if hasattr(editor, 'IsActive') and editor.IsActive():
                editor.editor.window.ReplaceSelection(text)
                return
        self.InsertStrToShell(text)

    def InsertStrToShell(self, text):
        "Insert text into the shell's command prompt"
        # clean text
        text = text.rstrip().replace('\r', '')

        lines = text.splitlines()
        n = len(lines)
        if n == 0:
            return
        elif n > 1:
            for i in xrange(1, n):
                lines[i] = '... ' + lines[i]
        text = os.linesep.join(lines)

        if not self.shell.CanEdit():
            pos = self.shell.GetLastPosition()
            self.shell.SetSelectionStart(pos)
            self.shell.SetSelectionEnd(pos)

        self.shell.ReplaceSelection(text)

    def OnAbout(self, event):
        """Display an About window."""
        about = AboutFrame(self)
        if wx.__version__ >= '2.9':
            about.ShowWithEffect(wx.SHOW_EFFECT_BLEND)
        else:
            about.Show()
        about.SetFocus()
        return about

    def OnActivate(self, event):
        # logging.debug(" Shell Activate Event: {0}".format(event.Active))
        if hasattr(self.shell, 'Destroy'):  # if alive
            if event.Active:
                self.shell.SetCaretForeground(wx.Colour(255, 0, 0))
                self.shell.SetCaretPeriod(500)
            else:
                self.shell.SetCaretForeground(wx.Colour(200, 200, 200))
                self.shell.SetCaretPeriod(0)

    def OnClearTerminal(self, event=None):
        self.shell.clear()
        self.shell.prompt()

    def OnClose(self, event):
        # http://stackoverflow.com/a/1055506/166700
        logging.debug("WxTerm Shell OnClose")
        if event.CanVeto():
            self.Hide()
            event.Veto()
        else:
            # make sure pdb is dectivated
            n = self.shell.GetLineCount()
            line = self.shell.GetLine(n - 1)
            if line.startswith('(Pdb) '):
                self.ExecCommand('exit')

            # shut down
            self.SaveSettings()
            event.Skip()

    def OnCloseAllPlots(self, event):
        self.CloseAllPlots()

    def OnComment(self, event):
        win = self.get_active_window()
        win.Comment()

    def OnCopy(self, event):
        win = wx.Window.FindFocus()
        if hasattr(win, 'Copy'):
            win.Copy()
            return

        win = self.get_active_window()
        if hasattr(win, 'canvas'):  # matplotlib figure
            if wx.TheClipboard.Open():
                try:
                    # save temporary pdf file
                    path = tempfile.mktemp('.pdf')
                    win.canvas.figure.savefig(path)
                    # copy path
                    do = wx.FileDataObject()
                    do.AddFile(path)
                    wx.TheClipboard.SetData(do)
                finally:
                    wx.TheClipboard.Close()
        else:
            logging.debug("can't copy: %s" % win)

    def OnCopyPlus(self, event):
        win = wx.Window.FindFocus()
        if hasattr(win, 'CopyWithPrompts'):
            win.CopyWithPrompts()
            return

        win = self.get_active_window()
        if hasattr(win, 'canvas'):  # matplotlib figure
            if wx.TheClipboard.Open():
                try:
                    # save temporary pdf file
                    path = tempfile.mktemp('.png')
                    win.canvas.figure.savefig(path)
                    # copy path
                    do = wx.FileDataObject()
                    do.AddFile(path)
                    wx.TheClipboard.SetData(do)
                finally:
                    wx.TheClipboard.Close()

    def OnDuplicate(self, event):
        win = self.get_active_window()
        Id = event.GetId()
        if Id == ID.DUPLICATE_WITH_OUTPUT:
            win.DuplicateFull()
        else:
            win.Duplicate()

    def OnDestroyIcon(self, evt):
        logging.debug("DESTROY ICON CALLED")
        self.eelbrain_icon.Destroy()
        evt.Skip()

    def OnExecFile(self, event=None):
        """
        Execute a file in the shell.

        filename: if None, will ask
        isolate: execute with separate globals, do not touch the shell's
                 globals

        """
        self.ExecFile()

    def OnFileClose(self, event):
        """
        Handler to catch and distribute 'close' command  in Os-X
        (see wx.py.frame.Frame)

        """
        win = self.get_active_window()
        if win:
            try:
                win.Close()
            except:
                win.Destroy()
                raise
        else:
            event.Skip()

    def OnExecDocument(self, event):
        win = self.get_active_window()
        win.ExecDocument()

    def OnExecDocumentFromDisk(self, event):
        win = self.get_active_window()
        win.ExecDocumentFromDisk()

    def OnExecSelection(self, event):
        win = self.get_active_window()
        win.ExecSelection()

    def OnFileNew(self, event=None):
        self.create_py_editor()

    def OnFileOpen(self, event=None, path=None):
        if path is None:
            path = ui.ask_file("Open File", "Open a Python script in an "
                               "editor, or attach pickled data",
                               [('Known Files (*.py, *.pickled)',
                                 '*.py;*.pickled')])
            if not path:
                return

        if isinstance(path, basestring) and os.path.isfile(path):
            _, ext = os.path.splitext(path)
            ext = ext.lower()
            if ext == '.py':
                self.create_py_editor(pyfile=path)
            elif ext == '.pickled':
                try:
                    with open(path, 'rb') as fid:
                        dinnerplate = pickle.load(fid)
                except Exception as exc:
                    msg = '%s: %s' % (type(exc).__name__, exc)
                    sty = wx.OK | wx.ICON_ERROR
                    wx.MessageBox(msg, "Unplicking Failed", style=sty)
                    raise exc

                typename = dinnerplate.__class__.__name__
                msg = ("What name should the unpickled data of\ntype %r be assigned "
                       "to?\n(Leave empty to attach it)" % typename)
                name0 = os.path.splitext(os.path.basename(path))[0]
                dlg = wx.TextEntryDialog(self, msg, "Name for Unpickled Content", name0)
                while True:
                    if dlg.ShowModal() == wx.ID_OK:
                        name = str(dlg.GetValue()).strip()
                        if name:
                            if is_py_varname(name):
                                msg = ">>> %s = pickle.load(open(%r))" % (name, path)
                                self.shell_message(msg, internal_call=True)
                                self.global_namespace[name] = dinnerplate
                                return
                            else:
                                msg = "%r is not a valid python variable name" % name
                                edlg = wx.MessageDialog(self, msg, "Invalid name",
                                                        wx.OK | wx.ICON_ERROR)
                                edlg.ShowModal()
                        else:
                            self.shell_message(">>> attach(pickle.load(open(%r)))" % path,
                                               internal_call=True)
                            self.attach(dinnerplate, _internal_call=True)
                            return
                    else:
                        return

            else:
                statinfo = os.stat(path)
                if statinfo.st_size < 100000:
                    self.create_py_editor(path)
                else:
                    msg = ("Error: %r is no known file extension. Since the "
                           "file is bigger than 100 kb, it was not read as "
                           "plain text." % ext)
                    dlg = wx.MessageDialog(self, msg, "Error LOoding File",
                                           wx.OK | wx.ICON_ERROR)
                    dlg.ShowModal()
        else:
            msg = "No valid file path: %r" % path
            dlg = wx.MessageDialog(self, msg, "Error", wx.OK | wx.ICON_ERROR)
            dlg.ShowModal()

    def OnFileSave(self, event):
        logging.debug("shell.OnFileSave()")
        win = self.get_active_window()
        if hasattr(win, 'bufferSave'):
            win.bufferSave()
        elif hasattr(win, 'toolbar') and hasattr(win.toolbar, 'save_figure'):
            # matplotlib figure
            win.toolbar.save_figure()
        else:
            event.Skip()

    def OnFileSaveAs(self, event):
        logging.debug("shell.OnFileSaveAs()")
        win = self.get_active_window()
        if hasattr(win, 'bufferSave'):
            win.bufferSave()
        elif hasattr(win, 'toolbar') and hasattr(win.toolbar, 'save_figure'):
            # matplotlib figure
            win.toolbar.save_figure()
        else:
            event.skip()

    def OnFocusPrompt(self, event):
        self.Show()
        self.shell.DocumentEnd()
        self.Raise()

    def OnHelp(self, event):
        "Help invoked through the Shell Menu"
        win = self.get_active_window()
        # check that the active window has a GetCurLine method
        if hasattr(win, 'GetCurLine'):
            text, pos = win.GetCurLine()
        else:
            dlg = wx.MessageDialog(self, "The active window does not have a "
                                   "GetCurLine function",
                                   "Help Call Failed", wx.OK | wx.ICON_ERROR)
            dlg.ShowModal()
            dlg.Destroy()
            return

        logging.debug("Shell: help called for : %s (%i)" % (text, pos))
        self.OnHelpViewer(topic_str=text, pos=pos)
        win.Raise()

    def OnHelpExternal(self, event):
        "Called from the Help menu to open external resources"
        Id = event.GetId()
        if Id == ID.HELP_EELBRAIN:
            webbrowser.open("https://pythonhosted.org/eelbrain/")
        elif Id == ID.HELP_MPL:
            webbrowser.open("http://matplotlib.org/api/pyplot_summary.html")
        elif Id == ID.HELP_MDP:
            webbrowser.open("http://mdp-toolkit.sourceforge.net/documentation.html")
        elif Id == ID.HELP_PDB:
            webbrowser.open("http://docs.python.org/library/pdb.html")
        elif Id == ID.HELP_PYTHON:
            webbrowser.open("http://docs.python.org/")
        else:
            raise ValueError("Invalid help ID")

    def OnHelpViewer(self, event=None, topic=None, topic_str=None, pos=None):
        """
        topic :
            object
        topic_str :
            string that can be evaluated to get topic object
        pos :
            With topic_str, the pos argument can be used to search for a
            Python object name within topic_str. When the user asks for help
            in the shell or editor, the topic_str is the current line and pos
            is the caret position.

        """
        if self.help_viewer:
            if self.help_viewer.IsShown():
                self.help_viewer.Raise()
            else:
                self.help_viewer.Show(True)
        else:
#            x, y, w, h = self.GetRect().Get()
            x_min, y_min, x_max, y_max = wx.Display().GetGeometry()
            width = 650
            x_pos = x_max - width  # min(x + w, x_max-600)
            y_pos = 22
            height = y_max - y_min - y_pos - 55  # min(800, y_max-y_min-21)
            wpos = (x_pos, y_pos)
            wsize = (width, height)
            logging.debug("help viewer: size=%s, pos=%s" % (wsize, wpos))
            self.help_viewer = HelpFrame(self, size=wsize, pos=wpos)
            self.help_viewer.Show()

        if topic_str and (pos is not None):
            # TODO: use RE
#                p = re.compile('')
            # find the python command around pos in the line
            topic_scan = topic_str.replace('.', 'a')
            start = pos
            n_max = len(topic_str)
            end = min(pos, n_max)
            while (end < n_max) and is_py_char(topic_scan[end]):
                end += 1
            while is_py_char(topic_scan[start - 1]) and start > 0:
                start -= 1
            topic_str = topic_str[start:end]

        if topic_str:
            self.help_viewer.text_lookup(topic_str)
        else:
            self.help_viewer.HelpLookup(topic)

    def OnInsertColor(self, event=None):
        ctup = event.GetValue()
        mplc = tuple([round(c / 256., 3) for c in ctup[:3]])
        self.InsertStr(str(mplc))

    def OnInsertPath_File(self, event):
        self.OnInsertPath(event, t=0)

    def OnInsertPath_Dir(self, event):
        self.OnInsertPath(event, t=1)

    def OnInsertPath_New(self, event):
        self.OnInsertPath(event, t=2)

    def OnInsertPath(self, event, t=None):
        if t is None:
            t = event.GetSelection()
        # get
        if t == 0:  # File
            filenames = ui.ask_file(mult=True)
        elif t == 1:  # Dir
            filenames = ui.ask_dir()
        elif t == 2:  # New
            filenames = ui.ask_saveas(title="Get New File Name",
                                       message="Please Pick a File Name",
                                       ext=None)

        string = droptarget.filename_repr(filenames)

        if string:  # insert into terminal
            self.InsertStr(string)

    def OnMaximize(self, event=None):
        logging.debug("SHELLFRAME Maximize received")
        self.Resize(ID.SIZE_MAX)

    def OnOpenHistory(self, event):
        id_ = event.GetId()
        history = self.shell.history
        if id_ == ID.OPEN_HISTORY_CURRENT:
            for i, item in enumerate(history):
                if item.startswith(history_session_hdr % ""):
                    history = history[:i + 1]
                    break
            txt = os.linesep.join(reversed(history))
            name = "Current Session History"
        elif id_ == ID.OPEN_HISTORY:
            txt = os.linesep.join(reversed(history))
            name = "History"
        else:
            name, txt = self._history_items[id_]

        editor = self.create_py_editor(name=name)
        editor.editor.window.ReplaceSelection(txt)

    def OnOpenMneGui(self, event):
        Id = event.GetId()
        target = 'g'
        if target in self.global_namespace:
            i = 0
            name = 'g%i'
            while name % i in self.global_namespace:
                i += 1
            target = name % i

        import mne
        if (('mne' not in self.global_namespace) or
            (self.global_namespace['mne'] is not mne)):
            self.ExecCommand('import mne')

        cmd = self.mne_cmds[Id]
        cmd = "%s = mne.gui.%s()" % (target, cmd)
        self.ExecCommand(cmd)

    def OnOpenWindowMenu(self, event):
        "Updates open windows to the menu"
        menu = event.GetMenu()
#        ID = event.GetMenuId() (is always 0)
        name = menu.GetTitle()
        if name == "&Window":
            logging.debug("Window Menu Open")
            # update names
            for ID, m in self.windowMenuMenuItems.iteritems():
                window = self.windowMenuWindows[ID]
                title = window.GetTitle()
                m.SetText(title)

    def OnPreferences(self, event=None):
        if self.preferencesDialog:
            self.preferencesDialog.Raise()
        else:
            dlg = PreferencesDialog(self)
            dlg.Show()
            dlg.CenterOnScreen()
            self.preferencesDialog = dlg

    def OnPyplotDraw(self, event=None):
        plt.draw()
        if plt.get_backend() == 'WXAgg':
            plt.show()

    def OnQuit(self, event=None):
        logging.debug("WxTerm Shell OnQuit")

        # close all windows
        for w in wx.GetTopLevelWindows():
            if w is not self:
                if not w.Close():
                    return

        if self.help_viewer:
            self.help_viewer.Close()
        self.CloseAllPlots()
        ui.kill_progress_monitors()

        self.Close(force=True)

    def OnRecentItemLoad(self, event):
        fileNum = event.GetId() - wx.ID_FILE1
        logging.debug("History Load: %s" % fileNum)
        path = self.filehistory.GetHistoryFile(fileNum)
        logging.debug("History Load: %s" % path)
        self.OnFileOpen(path=path)

    def OnResize(self, event):
        Id = event.GetId()
        self.Resize(Id)

    def OnSelectColourAlt(self, event=None):
        if hasattr(self, 'c_dlg'):
            if hasattr(self.c_dlg, 'GetColourData'):
                c = self.c_dlg.GetColourData()
                c = c.GetColour()
                self.shell.ReplaceSelection(str(c))
            else:
                del self.c_dlg
        if not hasattr(self, 'c_dlg'):
            self.c_dlg = wx.ColourDialog(self)

    def OnShowExamples(self, event=None):
        """
        __init__(self, Window parent, String message=FileSelectorPromptStr,
            String defaultDir=EmptyString, String defaultFile=EmptyString,
            String wildcard=FileSelectorDefaultWildcardStr,
            long style=FD_DEFAULT_STYLE,
            Point pos=DefaultPosition) -> FileDialog
        """
        dialog = wx.FileDialog(self, "Open Eelbrain Example",
                               defaultDir=self._examples_root,
                               wildcard="Python Scripts (*.py)|*.py",
                               style=wx.FD_OPEN)
        dialog.ShowModal()
        path = dialog.GetPath()
        if path:
            self.create_py_editor(pyfile=path)

    def OnTableNew(self, event=None):
        self.FrameTable(None)

    def OnTogglePyplotMgr(self, event=None):
        if self.P_mgr.IsShown():
            self.P_mgr.Show(False)
        else:
            self.P_mgr.Show()

    def OnUpdateMenu(self, event):
        """Replace form wx.py.frame.Frame to update new menu items and
        preferences dialog.
        """
        win = self.get_active_window()
        id_ = event.GetId()
        if id_ == ID.COMMENT:
            event.Enable(hasattr(win, 'Comment'))
        elif id_ == ID.DUPLICATE:
            event.Enable(hasattr(win, 'Duplicate'))
        elif id_ == ID.DUPLICATE_WITH_OUTPUT:
            event.Enable(hasattr(win, 'DuplicateFull'))
        elif id_ == ID.EXEC_SELECTION:
            if not isinstance(event.GetEventObject(), wx.Menu):
                return
            canexec = win.CanExec() if hasattr(win, 'CanExec') else True
            event.Enable(canexec and hasattr(win, 'ExecSelection'))
        elif id_ == ID.EXEC_DOCUMENT:
            if not isinstance(event.GetEventObject(), wx.Menu):
                return
            canexec = win.CanExec() if hasattr(win, 'CanExec') else True
            event.Enable(canexec and hasattr(win, 'ExecDocument'))
        elif id_ == ID.EXEC_DOCUMENT_FROM_DISK:
            if not isinstance(event.GetEventObject(), wx.Menu):
                return
            canexec = win.CanExec() if hasattr(win, 'CanExec') else True
            event.Enable(canexec and hasattr(win, 'ExecDocumentFromDisk'))
        else:
            if id_ == ID_SAVEHISTORY and self.preferencesDialog:
                state = event.IsChecked()
                self.preferencesDialog.checkboxSaveHistory.SetValue(state)
            elif id_ == ID_AUTO_SAVESETTINGS and self.preferencesDialog:
                state = event.IsChecked()
                self.preferencesDialog.checkboxSaveSettings.SetValue(state)
            elif (id_ == wx.ID_COPY and hasattr(win, 'canvas') and
                  hasattr(win.canvas, 'figure') and
                  hasattr(win.canvas.figure, 'savefig')):  # matplotlib figure
                event.Enable(True)
                return
            elif id_ == ID_COPY_PLUS:
                if (hasattr(win, 'canvas') and hasattr(win.canvas, 'figure')
                    and hasattr(win.canvas.figure, 'savefig')):  # matplotlib figure
                    event.Enable(True)
                    event.SetText("Cop&y as PNG \tCtrl+Shift+C")
                    return
                else:
                    event.SetText("Cop&y Plus \tCtrl+Shift+C")
            super(ShellFrame, self).OnUpdateMenu(event)

    def OnWindowMenuActivateWindow(self, event):
        ID = event.GetId()
        window = self.windowMenuWindows[ID]
        window.Show()
        window.Raise()

    def pos_for_new_window(self, size=(200, 400)):
        x, y, w, h = self.GetRect().Get()
        x_min, y_min, x_max, y_max = wx.Display().GetGeometry()
        e_x = x + w
        if e_x + size[0] > x_max:
            e_x = x_max - size[0]
        return (e_x, y_min + 26)

    def pyplot_draw(self):
        # update plots if mpl Backend does not do that automatically
        if mpl.get_backend() in ['WXAgg']:
            if len(plt._pylab_helpers.Gcf.get_all_fig_managers()) > 0:
                plt.draw()
                plt.show()

    def recent_item_add(self, path, t):
        "t is 'e' or 'pydoc'"
        self.filehistory.AddFileToHistory(path)
        self.filehistory.Save(self.config)
        self.config.Flush()
        self.recent_menu_update_icons()

#        path_list = self.recent_files[t]
#        menu = self.recent_menus[t]
#
#        if path in path_list:
#            i = path_list.index(path)
#            if i == 0:
#                return
#            path_list.pop(i)
#            menu_item = menu.GetMenuItems()[i]
#            menu.RemoveItem(menu_item)
#        else:
#            # remove old item
#            if len(path_list) > 10:
#                path_list.pop(-1)
#                n = menu.GetMenuItemCount()
#                old_menu_item = menu.GetMenuItems()[n-1]
#                menu.RemoveItem(old_menu_item)
#            # new item
#            if t == 'e':
#                id = ID.RECENT_LOAD_E
#            elif t =='pydoc':
#                id = ID.RECENT_LOAD_PYDOC
#            else:
#                raise ValueError("t == %s"%t)
#            name = os.path.basename(path)
#            menu_item = wx.MenuItem(menu, id, name, path)
#        menu.InsertItem(0, menu_item)
#        # path list
#        path_list.insert(0, path)
#        with open(self.recent_files_path, 'w') as file:
#            pickle.dump(self.recent_files, file)
    def recent_menu_update_icons(self):
        for item in self.recent_menu.GetMenuItems():
            text = item.GetItemLabelText()
            if text.endswith('.py'):
                item.SetBitmap(Icon('documents/pydoc'))
            elif text.endswith('startup'):
                item.SetBitmap(Icon('documents/pydoc-startup'))
            else:
                item.SetBitmap(Icon('documents/unknown'))

    def RemovePyEditor(self, editor):
        if editor in self.editors:
            self.editors.remove(editor)
        ID = editor.GetId()
        self.windowMenuWindows.pop(ID)
        self.windowMenuMenuItems.pop(ID)
        self.windowMenu.Remove(ID)

    def Resize(self, Id=ID.SIZE_MAX):
        self.Show()
        display = wx.Display()
        area = display.GetClientArea()
        if Id == ID.SIZE_MAX:
            width = min(area.width // 2, 700)
            rect = wx.Rect(area.left, area.top, width, area.height)
        elif Id == ID.SIZE_MAX_NOTITLE:
            width = min(area.width // 2, 700)
            rect = wx.Rect(area.left, 0, width, area.height + area.top)
        elif Id == ID.SIZE_MIN:
            rect = wx.Rect(area.left, 200, 350, 600)
        self.SetRect(rect)

    def set_debug_mode(self, redirect, logfile=None, write=True):
        """Change the debug mode (whether to show internal exceptions)

        Parameters
        ----------
        redirect : bool
            Redirect output for internal internal exceptions.
        logfile : None | str
            A file to redirect the output to. If None, open a window with
            the output.
        write : bool
            Save the new settings to the configuration file.

        Notes
        -----
        Turning off debug mode only takes effect after restarting the
        Application.
        """
        if write:
            self.config.WriteBool('Debug/Redirect', redirect)
            self.config.Write('Debug/Logfile', logfile or '')

        if redirect:
            app = wx.GetApp()
            app.RedirectStdio(logfile)

    def shell_message(self, message, sep=False, ascommand=False, endline=True,
                      internal_call=False):
        """
        Proxy for shell.writeOut method

        kwargs
        ------
        sep = False: adds linebreaks at the top and at the bottom of message

        ascommand = False: adds the prompt in front of the message to mimmick
                   command

        internal_call: notification that the call is made by the app mainloop
                       rather than form the shell

        """
        ls = os.linesep
        if message != ls:
            message = message.rstrip(ls)

        if ascommand:
            message = sys.ps1 + message
            endline = True

        if endline:
            if internal_call:
                message = message + ls
        elif sep:
            if message[0] != ls:
                message = ls + message
            if message[-1] != ls:
                message = message + ls
            if message[-2] != ls:
                message = message + ls

        if internal_call:  # ascommand:
            logging.debug("internal shell_msg: %r" % message)
            self.shell.start_exec()
            self.shell.writeOut(message)
            self.shell.end_exec()
        else:
            logging.debug("external shell_msg: %r" % message)
            print message

