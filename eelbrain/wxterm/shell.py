"""
supplies attach function to ui module


TODO: 
management of attached items 
 - keep dictionary that can be checked and restored
experiment management

"""

import sys 
import logging
import os
import string
import re
import cPickle as pickle
import webbrowser

#import wx
import wx.stc
import wx.py.shell
import wx.lib.colourselect

import matplotlib as mpl
import matplotlib.pyplot as plt


from eelbrain import __version__
from eelbrain import ui
#from eelbrain.signal_processing import Experiment, _extension
#from eelbrain.signal_processing import isexperiment, is_experiment_item#, isDataset
_extension = 'jkldfsa'

#import eelbrain.wxgui.viewers as viewers

import eelbrain.utils.print_funcs as print_funcs
from eelbrain import wxutils
from eelbrain.wxutils import Icon
from eelbrain.wxutils import droptarget

import ID
import about_dialog
import preferences_dialog
#from experiment_frame import ExperimentFrame
from help import HelpViewer
from table import TableFrame
import py_editor
import mpl_tools





# Attributes that are hidden from autocompletion. Should only be attributes 
# that the user never wants to retrieve from any module. Hence, at the moment 
# there are some top-level module names and their abbreviations.
hidden_attr = [
               'np', 'sp', 'scipy', 'mpl', 'P', 'plt',
               'wx',
               ]

_punctuation = string.punctuation.replace('.', '')


def is_py_char(char):
    return (char.isalnum() or char == '_')

def is_py_varname(name):
    a = re.match('^[a-zA-Z_]', name) 
    b = re.match('[a-zA-Z0-9_]', name)
    return a and b

# subclass Shell in order to set some custom properties
class Shell(wx.py.shell.Shell):
    exec_mode = 0       # counter to determine whether other objects than the shell itself are writing to writeOut
    has_moved = False   # keeps track whether any entity has written to the shell in exec_mode
#    def __init__(self, *args, **kwargs): # Leads to recursion crash
#        wx.py.shell.Shell.__init__(self, *args, **kwargs)
#        self.exec_mode = 0
#        self.push('from __future__ import print_function')
    def autoCompleteShow(self, command, offset = 0):
        """
        Display auto-completion popup list from wxPython, with an additional 
        mechanism for filtering out unwanted names:
         - names listed in the object's __hide__ attribute
         - names in hidden_attr (defined in this module)
        
        """
        ###### MY ADDITIONAL PRUNING MECHANISM
        #logging.debug(" AutoComplete for command {c}".format(c=str(command)))
        ###### MY end
        self.AutoCompSetAutoHide(self.autoCompleteAutoHide)
        self.AutoCompSetIgnoreCase(self.autoCompleteCaseInsensitive)
        options = self.interp.getAutoCompleteList(command,
                                includeMagic=self.autoCompleteIncludeMagic,
                                includeSingle=self.autoCompleteIncludeSingle,
                                includeDouble=self.autoCompleteIncludeDouble)
        if options:
            ###### MY ADDITIONAL PRUNING MECHANISM
            if command.endswith('.'):
                if any(item in options for item in ['__all__', '__hide__']):
                    # strip command preceding module name
                    for c in _punctuation+' ':
                        if c in command:
                            command = command.split(c)[-1]
                    # lookup
                    source = self.Parent.global_namespace
                    for name in command.split('.'):
                        if name:
                            source = source[name].__dict__
                    if '__all__' in options:
                        options = source['__all__']
                        options.sort(cmp=lambda x,y: cmp(x.lower(), y.lower()))
                    elif '__hide__' in options:
                        for item in source['__hide__']:
                            if item in options:
                                options.remove(item)
            #logging.debug(" AutoComplete: %s"%str(options))
            for item in hidden_attr:
                if item in options:
                    options.remove(item)
            #options = [item for item in options if item not in hidden_attr]
            ###### MY end
            options = ' '.join(options)
            self.AutoCompShow(offset, options)
    def writeOut(self, message):
        """
        
        sep = False: adds linebreaks at the top and at the bottom of message
        
        ascommand = False: adds the prompt in front of the message to mimmick
                   command
        """
#        logging.debug(" SHELL (%s): %s" % (['std','exec'][self.exec_mode], message))
        if self.exec_mode:
            message = unicode(message)
            message = self.fixLineEndings(message)
#            logging.debug("WRITEOUT: '%s'"%message)
                
            
            start = self.promptPosStart  # start des prompt >>>
            end = self.promptPosEnd      # end   des prompt >>>
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
            self.promptPosEnd = new_start #new_end
#            
#            self.SetCurrentPos(new_end)
#            self.SetAnchor(new_end)
        else:
#            wx.py.shell.Shell.writeOut(self, message)
#            logging.debug("NORMAL-WRITEOUT: '%s'"%message)
            self.write(message)
    def writeErr(self, message):
        # TODO: color=(0.93359375, 0.85546875, 0.3515625)
        ls = os.linesep
#        if message != ls:
#            message = message.replace(ls, ls+'!   ')
        message = message.replace('"""', '"') # some deprecation errors contain """ messing up shell coloration
        self.writeOut(message)#, sep=True)
    def start_exec(self):
        self.exec_mode += 1
    def end_exec(self):
        self.exec_mode -= 1
        if self.exec_mode == 0 and self.has_moved:
            self.prompt()
            self.has_moved = False
    
#    TODO: catch crashing startup script
#    def execStartupScript(self, startupScript):
#        """Execute the user's PYTHONSTARTUP script if they have one."""
#        if startupScript and os.path.isfile(startupScript):
#            text = 'Startup script executed: ' + startupScript
##            self.push('print %r; execfile(%r)' % (text, startupScript))
#            self.interp.startupScript = startupScript
#        else:
#            self.push('')



class ShellFrame(wx.py.shell.ShellFrame):
    bufferOpen = 1 # Dummy attr so that py.frame enables Open menu 
    bufferNew = 1  # same for New menu command
    bufferClose = 1  # same for Close menu command (handled by OnFileClose)
    def __init__(self, parent, namespace, *args, **kwargs):
        
        app = wx.GetApp()
        
    # --- set up PREFERENCES ---
        # http://wiki.wxpython.org/FileHistory
        self.wx_config = wx.Config("eelbrain", style=wx.CONFIG_USE_LOCAL_FILE)
        """
        Options
        -------
                
        use: config.Read(name, default_value)
             config.Write(name, value)
        """
        self.filehistory = wx.FileHistory(10)
        self.filehistory.Load(self.wx_config)
        
#        stdp = wx.StandardPaths.Get()
#        pref_p = wx.StandardPaths.GetUserConfigDir(stdp)
        
        
    # SHELL initialization
        # put my Shell into wx.py.shell and init
        wx.py.shell.Shell = Shell
        kwargs.update(locals=namespace, title='Eelbrain Shell')
        dataDir = self.wx_config.Read("dataDir")
        if os.path.exists(dataDir):
            kwargs['dataDir'] = dataDir
        else:
            title = "warning: dataDir does not exist"
            msg = ("dataDir at %r does not exist and will not be set. See "
                   "preferences to change the dataDir." % dataDir)
            ui.message(title, msg, '!')
        
        wx.py.shell.ShellFrame.__init__(self, parent, *args, **kwargs)
        self.SetStatusText('Eelbrain %s' % __version__)
        
        droptarget.set_for_strings(self.shell)
        
        # config
#        stdout = wx.py.shell.PseudoFileOut(self.shell_message)
#        wx.py.shell.sys.stdout = stdout
#        sys.stdout = stdout
#        self.shell.stdout = stdout
#        self.shell.autoCompleteIncludeSingle = False
#        self.shell.autoCompleteIncludeDouble = False
#        self.shell.autoCompleteIncludeMagic = True
        
        # attr
        self.global_namespace = namespace
        self.editors = []
        self.active_editor = None # editor last used; updated by Editor.OnActivate and Editor.__init__
        self.tables = []
        self.experiments = [] # keep track of ExperimentFrames
        self._attached_items = {}
        self.help_viewer = None
        
        
    # child windows
        x_min, y_min, x_max, y_max = wx.Display().GetGeometry()
        self.P_mgr = mpl_tools.PyplotManager(self, pos=(x_max-100, y_min+22+4))
        
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
        # add icons
        self.recent_menu_update_icons()
        
        self.Bind(wx.EVT_MENU_RANGE, self.OnRecentItemLoad, id=wx.ID_FILE1, id2=wx.ID_FILE9)

        # add menu item (can only add it to one menu apparently)
        help_txt = "Load an experiment or Python script from a list of recently used items"
        self.fileMenu.InsertMenu(0, wx.ID_ANY, 'Recent Files', recent_menu, help_txt)        
        
    # preferences menu
        if wx.Platform == '__WXMAC__':
            ID_PREFERENCES = app.GetMacPreferencesMenuItemId()
        else:  # although these seem to be the same -- 5022
            ID_PREFERENCES = wx.ID_PREFERENCES
        self.optionsMenu.Append(ID_PREFERENCES, "&Preferences...\tCtrl+,")
        self.Bind(wx.EVT_MENU, self.OnPreferences, id=wx.ID_PREFERENCES)
        
    # WINDOW MENU
        Id = get_menu_Id("&Window")
        if Id is None:
            m = self.windowMenu = wx.Menu()#title="&Window")
            m.SetTitle("&Window")
            Id_help = get_menu_Id('&Help')
            self.menuBar.Insert(Id_help, m, 'Window')
        else:
            m = self.windowMenu
        
        # name is used below in OnOpenWindowMenu to recognize Window menu
        self.Bind(wx.EVT_MENU_OPEN, self.OnOpenWindowMenu)
        
        m.Append(ID.P_MGR, "Pyplot Manager", 
                 "Toggle Pyplot Manager Panel.")#, wx.ITEM_CHECK)
        self.Bind(wx.EVT_MENU, self.OnTogglePyplotMgr, id=ID.P_MGR)
        
        m.Append(ID.PYPLOT_CLOSEALL, "Close All Plots",
                 "Closes all Matplotlib plots.")
        self.Bind(wx.EVT_MENU, self.OnP_CloseAll, id=ID.PYPLOT_CLOSEALL)
        
        # shell resizing
        m.AppendSeparator()
        m.Append(ID.SHELL_Maximize, 'Maximize Shell', 
                 "Expand the Shell to use the full screen height.")
        self.Bind(wx.EVT_MENU, self.OnResize_Max, id=ID.SHELL_Maximize)
        
        m.Append(ID.SHELL_HalfScreen, 'Half Screen Shell',
                 "Expand Shell to use the left half of the screen")
        self.Bind(wx.EVT_MENU, self.OnResize_HalfScreen, id=ID.SHELL_HalfScreen)
        
        m.Append(ID.SHELL_Window, 'Window Shell',
                 "Resize the Shell to window with standard width.")
        self.Bind(wx.EVT_MENU, self.OnResize_Win, id=ID.SHELL_Window)
        
        m.Append(ID.SHELL_Mini, 'Mini-Shell', "Resize the Shell to a small "
                 "window (e.g. for use as pocket calculator).")
        self.Bind(wx.EVT_MENU, self.OnResize_Min, id=ID.SHELL_Mini)
        
        # section with all open windows
        m.AppendSeparator()
        item = m.Append(self.GetId(), "Shell", "Bring shell to the front.")
        self.Bind(wx.EVT_MENU, self.OnWindowMenuActivateWindow, item)
        self.windowMenuWindows = {self.GetId(): self}
        self.windowMenuMenuItems = {}
        
        # clear shell: this can be done with edit->empty buffer
                
        
    # INSERT MENU
        m = self.insertMenu = wx.Menu()
        m.Append(ID.INSERT_Color, "Color",
                    "Insert color as (r, g, b)-tuple.")
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
        
    # HELP menu
        self.helpMenu.AppendSeparator()
        
        self.helpMenu.Append(ID.HELP_PYTHON, "Eelbrain (web)",
                             "Open the Eelbrain documentation pages in an external browser.")
        self.Bind(wx.EVT_MENU, self.OnHelpExternal, id=ID.HELP_EELBRAIN)
        
        self.helpMenu.Append(ID.HELP_PYTHON, "Python (web)",
                             "Open the official Python documentation page in an external browser.")
        self.Bind(wx.EVT_MENU, self.OnHelpExternal, id=ID.HELP_PYTHON)
        
        self.helpMenu.Append(ID.HELP_MPL, "Matplotlib (web)", 
                             "Open the Matplotlib homepage in an external browser.")
        self.Bind(wx.EVT_MENU, self.OnHelpExternal, id=ID.HELP_MPL)
        
        self.helpMenu.Append(ID.HELP_MDP, "mdp (web)", 
                             "Open the mdp Documentation page in an external browser.")
        self.Bind(wx.EVT_MENU, self.OnHelpExternal, id=ID.HELP_MDP)
        
        
    # --- TOOLBAR ---
        tb = self.CreateToolBar(wx.TB_HORIZONTAL)
        tb.SetToolBitmapSize(size=wx.Size(32,32))
        
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
        
        tb.AddLabelTool(ID.PYDOC_EXEC, "Exec", Icon("documents/pydoc-openrun"),
                        shortHelp="Run an existing Python script (without "
                        "opening it in an editor)")
        self.Bind(wx.EVT_TOOL, self.OnExecFile, id=ID.PYDOC_EXEC)

#        tb.AddLabelTool(ID.EXPERIMENT_NEW, "New", Icon("documents/experiment-new"),
#                        shortHelp="Create a new experiment")
#        self.Bind(wx.EVT_TOOL, self.OnNewExperiment, id=ID.EXPERIMENT_NEW) 
        
        tb.AddLabelTool(wx.ID_NEW, "New .Py Document", Icon("documents/pydoc-new"),
                        shortHelp="Open a new Python script editor")

        tb.AddLabelTool(ID.TABLE, "Table", Icon("documents/table"),
                        shortHelp="Open a new table")
        self.Bind(wx.EVT_TOOL, self.OnTableNew, id=ID.TABLE)
        tb.AddSeparator()
        
        # inserting (these controls have no shortHelp)
        fn_ctrl = wx.Choice(tb, wx.ID_FIND, (100,50), choices=['File', 'Dir', 'New'])
        self.Bind(wx.EVT_CHOICE, self.OnInsertPath, id=wx.ID_FIND)
        tb.AddControl(fn_ctrl)
        b = wx.lib.colourselect.ColourSelect(tb, -1, "Color", (0, 255, 0),
                                             size=wx.DefaultSize)
        b.Bind(wx.lib.colourselect.EVT_COLOURSELECT, self.OnInsertColor)
        self.color_selector = b
        # bind the 'insert color' menu entry
        self.Bind(wx.EVT_MENU, self.color_selector.Command, id=ID.INSERT_Color)
        tb.AddControl(b)
        #tb.AddLabelTool(ID.COLOUR_CHOOSER, "Color", Icon('apps.preferences-desktop-locale'))
        #self.Bind(wx.EVT_TOOL, self.OnSelectColourAlt, id=ID.COLOUR_CHOOSER)
        
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
        self.global_namespace['printdict'] = print_funcs.printdict
        self.global_namespace['printlist'] = print_funcs.printlist
        
        # other Bindings
        self.Bind(wx.EVT_MAXIMIZE, self.OnMaximize)
        self.Bind(wx.EVT_ACTIVATE, self.OnActivate)
        self.shell.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)        
        # my shell customization
        self.SetColours(self.shell)
        self.OnResize_HalfScreen(None)
        # icon
        if sys.platform != 'darwin':
            self.eelbrain_icon = Icon('eelbrain', asicon=True)
            self.SetIcon(self.eelbrain_icon)
            self.Bind(wx.EVT_CLOSE, self.OnDestroyIcon)
        
        
        # add help text from wx.py.shell
        text = wx.py.shell.HELP_TEXT
        self.__doc__ = text
    
    def attach(self, dictionary, _internal_call=False):
        """
        Adds a dictionary to the globals and keeps track of the items so that 
        they can be removed safely with the detach function.
        
        """
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
                attach = dict((k,v) for k,v in dictionary.iteritems() if k not in present)
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
    
    def create_py_editor(self, pyfile=None):
        """
        Creates and returns a new py_editor object.
        
        :arg pyfile: filename as string, or True in order to display an "open
            file" dialog (None creates an empty editor)
        :arg openfile: True if an 'open file' dialog should be shown after the
            editor is cerated
        
        """
        editor = py_editor.Editor(self, self, pos=self.pos_for_new_window(),
                                  pyfile=pyfile)
        
        editor.Show()
        self.editors.append(editor)
        
        # add to window menu
        ID = editor.GetId()
        m = self.windowMenu.Append(ID, "a", "Bring window to the front.")
        self.Bind(wx.EVT_MENU, self.OnWindowMenuActivateWindow, m)
        self.windowMenuWindows[ID] = editor
        self.windowMenuMenuItems[ID] = m
        
        return editor
    
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
        
        for k,v in detach.iteritems():
            if id(self.global_namespace[k]) == id(v):
                del self.global_namespace[k]
    
    def ExecFile(self, filename, shell_globals=True):
        """
        Execute a file in the shell.
        
        shell_globals determines wheter the shell's globals are submitted to
        the call to execfile or not.
        
        
        (!)
        A problem currently (also with the commented-out version below) is that
        __file__ and sys.argv[0] point to eelbrain.__main__ instead of the 
        executed file.  
        
        """
        if filename and os.path.exists(filename):
            os.chdir(os.path.dirname(filename))
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
#            save_stdout = sys.stdout
#            save_stderr = sys.stderr
#            save_stdin = sys.stdin
#
#            sys.stdout = self.shell.interp.stdout
#            sys.stderr = self.shell.interp.stderr
#            sys.stdin = self.shell.interp.stdin
#            
#            self.shell.start_exec()
#            execfile(filename)
#            self.shell.end_exec()
#            
#            sys.stdout = save_stdout
#            sys.stderr = save_stderr
#            sys.stdin = save_stdin
#            self.shell.setFocus()
        else:
            logging.error("shell.ExecFile: invalid filename (%r)"%filename)
    
    def ExecText(self, txt, out=False, title="unknown source", comment=None,
                 shell_globals=True, filedir=None, internal_call=False):
        """
        Compile txt and Execute it in the shell.
        
        kwargs
        ------ 
        out: shell.Execute
        title: is displayed in the shell and should identify the source of the 
            code. 
        comment: displayed after title
        shell_globals: determines wheter the shell's globals are submitted to
            the call to execfile or not.
        filedir: perform os.chdir before executing
        
        """
        if comment is None:
            msg = 'exec <%r>' % title
        else:
            msg = 'exec <%r, %s>' % (title, comment)
        self.shell_message(msg, ascommand=True, internal_call=internal_call)
        
        if filedir:
            os.chdir(filedir)
        
        if out:
            self.shell.Execute(txt)
        else:
            self.shell.start_exec()
            
            # this does not work
#            self.shell.interp.runsource(txt)
            
            # prepare txt
            txt = self.shell.fixLineEndings(txt)
            txt += os.linesep
            
            # 1
            cmd = "exec(compile(r'''{txt}''', '{title}', 'exec'), {globals})"
            if shell_globals:
                exec_globals = "globals()"
            else:
                exec_globals = '{}'
            
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
        return None
    
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
        self.shell.ReplaceSelection(text)
    
    def OnAbout(self, event):
        """Display an About window."""
        about = about_dialog.AboutFrame(self)
        if wx.__version__ >= '2.9':
            about.ShowWithEffect(wx.SHOW_EFFECT_BLEND)
        else:
            about.Show()
        about.SetFocus()
        return about

    def OnActivate(self, event=None):
        #logging.debug(" Shell Activate Event: {0}".format(event.Active))
        if hasattr(self.shell, 'Destroy'): # if alive
            if event.Active:
                self.shell.SetCaretForeground(wx.Colour(255,0,0))
                self.shell.SetCaretPeriod(500)
            else:
                self.shell.SetCaretForeground(wx.Colour(200,200,200))
                self.shell.SetCaretPeriod(0)
    
    def OnClearTerminal(self, event=None):
        self.shell.clear()
        self.shell.prompt()
    
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
        dialog = wx.FileDialog(self, style=wx.FD_OPEN)
        dialog.SetMessage("Select Python File")
        dialog.SetWildcard("Python files (*.py)|*.py")
        if dialog.ShowModal():
            filename = dialog.GetPath()
        self.ExecFile(filename)
    
    def OnFileClose(self, event):
        """
        Handler to catch and distribute 'close' command  in Os-X 
        (see wx.py.frame.Frame)
        
        """
        win = self.get_active_window()
        if win:
            if win is self:
                pass
            else:
                win.Close()
        else:
            event.Skip()
    
    def OnFileNew(self, event=None):
        self.create_py_editor()
    
    def OnFileOpen(self, event=None, path=None):
        if path is None:
            path = ui.ask_file(title="Open File",
                               message="Open a Python script in an editor, or attach pickled data", 
                               ext=[('py', 'Python script'), 
                                    ('pickled', 'Pickled data')])
            if not path:
                return
        
        if isinstance(path, basestring) and os.path.isfile(path):
            _, ext = os.path.splitext(path.lower())
            if ext == '.py':
                self.create_py_editor(pyfile=path)
            elif ext == '.pickled':
                with open(path) as FILE:
                    dinnerplate = pickle.load(FILE)
                
                msg = ("What name should the unpickled data be assigned "
                       "to? (Leave empty to attach it.)")
                dlg = wx.TextEntryDialog(self, msg, "Name?", "")
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
                msg = "Error: %r is no known file extension." % ext
                dlg = wx.MessageDialog(self, msg, "Error", wx.OK | wx.ICON_ERROR)
                dlg.ShowModal()
        else:
            msg = "No valid file path: %r" % path
            dlg = wx.MessageDialog(self, msg, "Error", wx.OK | wx.ICON_ERROR)
            dlg.ShowModal()
    
    def OnFindPath(self, event=None):
        filenames = ui.ask_file(wildcard='', mult=True)
        if filenames:
            if len(filenames) == 1:
                filenames = '"'+filenames[0]+'"'
            else:
                filenames = str(filenames)
            self.shell.ReplaceSelection(filenames)
    
    def OnHelp(self, event):
        "Help invoked through the Shell Menu"
        win = self.get_active_window()
        # check that the active window has a GetCurLine method
        if hasattr(win, 'GetCurLine'):
            text, pos = win.GetCurLine()
        else:
            dlg = wx.MessageDialog(self, "The active window does not have a "
                                   "GetCurLine function",
                                   "Help Call Failed", wx.OK|wx.ICON_ERROR)
            dlg.ShowModal()
            dlg.Destroy()
            return
        
        logging.debug("Shell: help called for : %s (%i)" % (text, pos))
        self.OnHelpViewer(topic_str=text, pos=pos)
    
    def OnHelpExternal(self, event):
        "Called from the Help menu to open external resources"
        Id = event.GetId()
        if Id == ID.HELP_EELBRAIN:
            webbrowser.open("http://christianmbrodbeck.github.com/Eelbrain/")
        elif Id == ID.HELP_MPL:
            webbrowser.open("http://matplotlib.sourceforge.net/")
        elif Id == ID.HELP_MDP:
            webbrowser.open("http://mdp-toolkit.sourceforge.net/documentation.html")
        elif Id == ID.HELP_PYTHON:
            webbrowser.open("http://docs.python.org/")
        else:
            raise ValueError("Invalid help ID")
    
    def OnHelpViewer(self, event=None, topic=None, topic_str=None, pos=None):
        """
        topic: object
        topic_str: string that can be evaluated to get topic object
        pos: With topic_str, the pos argument can be used to search for a
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
            hx = x_max-650 #min(x + w, x_max-600)
            hw = 650
            hh = y_max-y_min-22 # min(800, y_max-y_min-21)
            hy = 22
            logging.debug("help viewer: size=(%s,%s), pos=(%s,%s)"%(hw,hh,hx,hy))
            self.help_viewer = HelpViewer(self, size=(hw,hh), pos=(hx,hy))
            self.help_viewer.Show()
        
        if topic_str:
            if pos is not None:
                # find the python command around pos in the line
                topic_scan = topic_str.replace('.', 'a')
                start = pos
                n_max = len(topic_str)
                end = min(pos, n_max)
                while (end < n_max) and is_py_char(topic_scan[end]):
                    end += 1
                while is_py_char(topic_scan[start-1]) and start > 0:
                    start -= 1
                topic_str = topic_str[start:end]
            self.help_viewer.text_lookup(topic_str)
        else:
            self.help_viewer.Help_Lookup(topic)
    
    def OnInsertColor(self, event=None):
        ctup = event.GetValue()
        mplc = tuple([round(c/256., 3) for c in ctup[:3]])
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
        if t == 0: # File
            filenames = ui.ask_file(mult=True)
        elif t == 1: # Dir
            filenames = ui.ask_dir()
        elif t == 2: # New
            filenames = ui.ask_saveas(title = "Get New File Name",
                                      message = "Please Pick a File Name", 
                                      ext = None)
        # put to terminal
        if filenames:
            if len(filenames) == 1:
                filenames = '"'+filenames[0]+'"'
            else:
                filenames = '"'+str(filenames)+'"'
            
            if 'wxMSW' in wx.PlatformInfo:
                filenames = 'r'+filenames
            
            self.InsertStr(filenames)
    
    def OnKeyDown(self, event):
        key = event.GetKeyCode()
        mod = wxutils.key_mod(event)
#        logging.debug("app.shell.OnKeyDown(): key=%r, mod=%r" % (key, mod))
        
        if key == 68: # 'd'
            if mod == [1, 0, 1]: # alt -> also include output
                mod = [1, 0, 0]
                duplicate_output = True
            else:
                duplicate_output = False
        # copy selection to first editor
        if mod == [1, 0, 0]: # [command]
            if key==80: # 'p'
                plt.draw()
                if plt.get_backend() == 'WXAgg':
                    plt.show()
            elif key==68: # 'd'
                # make sure we have a target editor
                if not hasattr(self.active_editor, 'InsertLine'):
                    self.OnFileNew(event)
                    self.active_editor = self.editors[-1]
                editor = self.active_editor
                # prepare text for transfer
                text = self.shell.GetSelectedText()
                lines = [line for line in text.split(os.linesep) if len(line)>0]
                # FIXME: alt
                for line in lines:
                    line_stripped = self.shell.lstripPrompt(line)
                    if duplicate_output or len(line_stripped) < len(line):
                        try:
                            editor.InsertLine(line_stripped)
                        except:
                            logging.debug("'Duplicate to Editor' failed")
            else:
                event.Skip()
        else:
#            logging.info("shell key: %s"%key)
            event.Skip()
    
    def OnMaximize(self, event=None):
        logging.debug("SHELLFRAME Maximize received")
        self.OnResize_Max(event)
    
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
        dlg = preferences_dialog.PreferencesDialog(self)
#        dlg = wx.MessageDialog(self, "Test", 'test', wx.OK)
        dlg.Show()
#        dlg.Destroy()
    
    def OnP_CloseAll(self, event=None):
        plt.close('all')
    
    def OnQuit(self, event=None):
        logging.debug(" QUIT")
        if self.help_viewer:
            self.help_viewer.Close()
        unsaved = []
        for ed in self.editors:
            if hasattr(ed, 'editor'): # check if alife
                if hasattr (ed.editor, 'hasChanged'): # editor without doc
                    if ed.editor.hasChanged():
                        unsaved.append(ed)
                    else:
                        ed.Close()
        if len(unsaved) == 1:
            unsaved[0].OnClose(event)
        elif len(unsaved) > 0:
            txt = '\n'.join([u.Title for u in unsaved])
            msg = wx.MessageDialog(None, txt, "Review Unsaved Py-Docs?", 
                                   wx.ICON_QUESTION|wx.YES_NO|wx.YES_DEFAULT|\
                                   wx.CANCEL)
            command = msg.ShowModal()
            if command == wx.ID_CANCEL:
                return
            else:
                for ed in unsaved:
                    if command == wx.ID_YES:
#                        ed.bufferSave()
                        ed.Close()
                    else:
                        ed.Destroy()
        self.OnP_CloseAll()
        self.Close()
    
    def OnRecentItemLoad(self, event):
        fileNum = event.GetId() - wx.ID_FILE1
        logging.debug("History Load: %s"%fileNum)
        path = self.filehistory.GetHistoryFile(fileNum)
        logging.debug("History Load: %s"%path)
        self.OnFileOpen(path=path)
    
    def OnResize_Max(self, event):
        x_min, y_min, x_max, y_max = wx.Display().GetGeometry()
        x_size = min(x_max-x_min, 800) + x_min
        self.SetPosition((x_min, y_min))
        self.SetSize((x_size, y_max))
    
    def OnResize_HalfScreen(self, event):
        x_min, y_min, x_max, y_max = wx.Display().GetGeometry()
        self.SetPosition((x_min, y_min))
        x_size = min(x_max//2, 800) - x_min
        y_size = y_max - y_min
        self.SetSize((x_size, y_size))
    
    def OnResize_Win(self, event):
        x_min, y_min, x_max, y_max = wx.Display().GetGeometry()
        self.SetPosition((x_min, y_min+50))
        x_size = 800 + x_min
        y_size = y_max - y_min - 100
        self.SetSize((x_size, y_size))
    
    def OnResize_Min(self, event):
        self.SetPosition((50, 200))
        self.SetSize((350, 600))   
         
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
    
    def OnTableNew(self, event=None):
        self.FrameTable(None)
    
    def OnTogglePyplotMgr(self, event=None):
        if self.P_mgr.IsShown():
            self.P_mgr.Show(False)
        else:
            self.P_mgr.Show()
    
    def OnWindowMenuActivateWindow(self, event):
        ID = event.GetId()
        window = self.windowMenuWindows[ID]
        window.SetFocus()
        
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
        self.filehistory.Save(self.wx_config)
        self.wx_config.Flush()
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
        py_icon = Icon('documents/pydoc')
        e_icon = Icon('documents/experiment')
        for item in self.recent_menu.GetMenuItems():
            text = item.GetItemLabelText()
            if text.endswith('.py'):
                item.SetBitmap(py_icon)
            elif text.endswith(_extension):
                item.SetBitmap(e_icon)
        
    def RemovePyEditor(self, editor):
        if editor in self.editors:
            self.editors.remove(editor)
        ID = editor.GetId()
        self.windowMenuWindows.pop(ID)
        self.windowMenuMenuItems.pop(ID)
        self.windowMenu.Remove(ID)
        
    def SetColours(self, obj):
        obj.SetCaretWidth(2)
        if 'wxMac' in wx.PlatformInfo:
            # --> wx.py.shell.editwindow.EditWindow.__config()
            FACES = { 'times'     : 'Lucida Grande',
                      'mono'      : 'Courier New',#'Monaco',
                      'helv'      : 'Geneva',
                      'other'     : 'new century schoolbook',
                      'size'      : 12,
                      'lnsize'    : 16,
                      'backcol'   : '#F0F0F0',#'#000010',
                      'calltipbg' : '#101010',
                      'calltipfg' : '#F09090',
                    }
            obj.StyleClearAll()
            obj.setStyles(FACES)
            obj.CallTipSetBackground(FACES['calltipbg'])
            obj.CallTipSetForeground(FACES['calltipfg'])
        if True:
            obj.StyleSetSpec(wx.stc.STC_STYLE_DEFAULT,
                          "fore:#FFFF00")#,back:#666666")
            obj.StyleSetSpec(wx.stc.STC_P_DEFAULT,  # '\'
                          "fore:#000000")
            obj.StyleSetSpec(wx.stc.STC_P_NUMBER,
                          "fore:#0000FF")
            obj.StyleSetSpec(wx.stc.STC_STYLE_CONTROLCHAR,
                          "fore:#FF0000")
            obj.StyleSetSpec(wx.stc.STC_P_STRING,
                          "fore:#FF0000")
            obj.StyleSetSpec(wx.stc.STC_P_CHARACTER,
                          "fore:#0000FF")
            obj.StyleSetSpec(wx.stc.STC_P_OPERATOR,
                          "fore:#000010")
            obj.StyleSetSpec(wx.stc.STC_P_DECORATOR,
                          "fore:#FF00FF")
            obj.StyleSetSpec(wx.stc.STC_P_WORD,
                          "fore:#FF00FF")
            obj.StyleSetSpec(wx.stc.STC_STYLE_MAX,
                          "fore:#0000FF")
            #obj.SetSelForeground(True, wx.Colour(255,255,255))
    
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
        
        if internal_call:# ascommand:
            logging.debug("internal shell_msg: %r" % message)
            self.shell.start_exec()
            self.shell.writeOut(message)
            self.shell.end_exec()
        else:
            logging.debug("external shell_msg: %r" % message)
            print message

