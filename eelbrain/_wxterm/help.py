"""
Help Viewer

TODO: use wx.html2

"""

import inspect
import logging
import types
import webbrowser

import wx.html
import docutils.core

from .. import fmtxt
from .._wxutils import Icon, ID



HtmlTemplate = """<pre class="literal-block">%s</pre>"""


def rst2html(rst):
    """
    reStructuredText to HTML parsing following:
    http://stackoverflow.com/questions/6654519
    """
    # remove leading whitespaces; make an exception for the first line,
    # since several functions start their docstring on the first line
    rst = fmtxt.unindent(rst, True)

    try:
        html = docutils.core.publish_parts(rst, writer_name='html')['body']
        if "ERROR/3" in html:
            raise RuntimeError("rst2html unresolved cross-ref")
        logging.debug("rst2html success")
#            html = os.linesep.join((html['stylesheet'], html['body']))
#            html = html['whole']
#            html = '<span style="color: rgb(0, 0, 255);">RST2HTML:</span><br>' + html
    except:
        html = HtmlTemplate % rst
        logging.debug("rst2html failed")
#            html = '<span style="color: rgb(255, 0, 0);">RST2HTML FAILED:</span><br>' + html
    return html


def doc2html(obj, default='No doc-string.<br>'):
    """
    Returns the object's docstring as html, or the default value if there is
    no docstring.

    """
    if hasattr(obj, '__doc__') and isinstance(obj.__doc__, basestring):
        txt = rst2html(obj.__doc__)
    else:
        txt = default
    return txt




def format_chapter(title, txt):
    return '\n'.join(('\n',
                      '-' * 80,
                      title + ' |',
                      '-' * (2 + len(title)),
                      txt, '\n'))

def format_subtitle(subtitle):
    return '<span style="color: rgb(102, 102, 102);">%s</span>' % subtitle


class HelpFrame(wx.Frame):
    def __init__(self, parent, *args, **kwargs):
        wx.Frame.__init__(self, parent, *args, **kwargs)
        self.EnableCloseButton(False)
        self.parent_shell = parent

        self.help_formatter = HTMLHelpFormatter()

        style = wx.NO_FULL_REPAINT_ON_RESIZE
        self.HTMLPanel = wx.html.HtmlWindow(self, style=style)
        self.HTMLPanel.Bind(wx.html.EVT_HTML_LINK_CLICKED, self.OnLinkClicked)

#        html.SetRelatedFrame(parent)
        # after wxPython Demo
        if "gtk2" in wx.PlatformInfo:
            self.SetStandardFonts()


        # prepare data container
        self.history = []
        self.current_history_id = -1

        # TOOLBAR ---
        self.toolbar = tb = self.CreateToolBar(wx.TB_HORIZONTAL)
        tb.SetToolBitmapSize(size=(32, 32))

        # hide
        tb.AddLabelTool(ID.HELP_HIDE, "Hide", Icon("tango/status/image-missing"))
        self.Bind(wx.EVT_TOOL, self.OnHide, id=ID.HELP_HIDE)
        tb.AddSeparator()

        # forward/backward
        tb.AddLabelTool(wx.ID_HOME, "Home", Icon("tango/places/start-here"))
        self.Bind(wx.EVT_TOOL, self.OnHome, id=wx.ID_HOME)
        tb.AddLabelTool(wx.ID_BACKWARD, "Back", Icon("tango/actions/go-previous"))
        self.Bind(wx.EVT_TOOL, self.OnBackward, id=wx.ID_BACKWARD)
        tb.AddLabelTool(wx.ID_FORWARD, "Next", Icon("tango/actions/go-next"))
        self.Bind(wx.EVT_TOOL, self.OnForward, id=wx.ID_FORWARD)
        tb.EnableTool(wx.ID_FORWARD, False)
        tb.EnableTool(wx.ID_BACKWARD, False)

        # text search
        self.history_menu = wx.Menu()
        item = self.history_menu.Append(-1, "Help History")
        item.Enable(False)
        search_ctrl = wx.SearchCtrl(tb, wx.ID_HELP, style=wx.TE_PROCESS_ENTER,
                                    size=(300, -1))
        search_ctrl.Bind(wx.EVT_TEXT_ENTER, self.OnSelfSearch)
        search_ctrl.Bind(wx.EVT_SEARCHCTRL_SEARCH_BTN, self.OnSelfSearch)
        search_ctrl.SetMenu(self.history_menu)
        self.history_menu.Bind(wx.EVT_MENU, self.OnSearchhistory)
        search_ctrl.ShowCancelButton(True)
        self.Bind(wx.EVT_SEARCHCTRL_CANCEL_BTN, self.OnSearchCancel, search_ctrl)
        tb.AddControl(search_ctrl)
        self.search_ctrl = search_ctrl


        if wx.__version__ >= '2.9':
            tb.AddStretchableSpace()
        else:
            tb.AddSeparator()

        # clear cache
        tb.AddLabelTool(ID.CLEAR_CACHE, "Clear Cache", Icon("tango/actions/edit-clear"))
        self.Bind(wx.EVT_TOOL, self.OnClearCache, id=ID.CLEAR_CACHE)

        # finish
        tb.Realize()

    def display(self, name):
        if name in self.help_formatter.help_items:
            content = self.help_formatter.help_items[name]
            self.HTMLPanel.SetPage(content)
        else:
            raise ValueError("No help object for %r" % name)

        self.search_ctrl.SetValue(name)
        self.SetTitle("Help: %s" % name)
        self.Show()

    def HelpLookup(self, topic=None, name=None):
        """
        Display help for a topic. Topic can be
         - None -> display default help
         - an object -> display help for the object based on its doc-string

        """
        if topic is None:
            name = 'Start Page'
        else:
            name = self.help_formatter.add_object(topic)

        self.display(name)

        if name in self.history:
            index = self.history.index(name)
            self.history.pop(index)

        i = self.current_history_id
        if (i != -1) and (len(self.history) > i + 1):
            self.history = self.history[0:i + 1]

        self.history.append(name)
        self.set_current_history_id(-1)

        self.Raise()

    def OnClearCache(self, event):
        self.help_formatter.delete_cache()

    def OnHide(self, event=None):
        self.Show(False)

    def OnHome(self, event=None):
        self.HelpLookup(topic=None)

    def OnBackward(self, event=None):
        i = self.current_history_id
        if i == -1:
            i = len(self.history) - 1

        i -= 1
        name = self.history[i]
        self.display(name)
        self.set_current_history_id(i)

    def OnForward(self, event=None):
        i = self.current_history_id + 1

        name = self.history[i]
        self.display(name)
        self.set_current_history_id(i)

    def OnLinkClicked(self, event):
        URL = event.GetLinkInfo().GetHref()
        # is there a better way to distinguish external from internal links?
        if URL.startswith('http://') or URL.startswith('www.'):
            webbrowser.open(URL)
        else:
            event.Skip()

    def OnSearchCancel(self, event=None):
        self.search_ctrl.Clear()

    def OnSearchhistory(self, event=None):
        i = event.GetId() - 1
        self.display(i)

    def OnSelfSearch(self, event=None):
        txt = event.GetString()
        if len(txt) > 0:
            self.text_lookup(txt)

    def set_current_history_id(self, i):
        self.current_history_id = i

        if i == -1:
            exists_greater = False
            exists_smaller = len(self.history) > 1
        else:
            exists_greater = len(self.history) > i + 1
            exists_smaller = i > 0
        self.toolbar.EnableTool(wx.ID_FORWARD, exists_greater)
        self.toolbar.EnableTool(wx.ID_BACKWARD, exists_smaller)

    def text_lookup(self, txt):
        logging.debug("Help text_lookup: %r" % txt)
        try:
            obj = eval(txt, self.parent_shell.global_namespace)
        except:
            dlg = wx.MessageDialog(self, "No object named %r in shell namespace" % txt,
                                   "Help Lookup Error:", wx.OK | wx.ICON_ERROR)
            dlg.ShowModal()
            dlg.Destroy()
        else:
            self.HelpLookup(obj, name=txt)


class HTMLHelpFormatter(object):
    """
    """
    def __init__(self):
        self.delete_cache()

    def add_object(self, obj, name=None):
        if not name:
            if hasattr(obj, '__name__'):
                name = obj.__name__
            elif hasattr(obj, '__class__'):
                name = obj.__class__.__name__
            else:
                raise ValueError("No Name For Help Object")

        if name not in self.help_items:
            self.help_items[name] = self.parse_object(obj)

        return name

    def delete_cache(self):
        "removes all stored help entries"
        self.help_items = {'Start Page': self.get_help_home()}

    def get_help_home(self):
        title = "PyShell"
        title = '\n'.join([title, '-' * len(title)])
        intro = "PyShell ``HELP_TEXT``::"
        pyshell_doc = '\t' + wx.py.shell.HELP_TEXT.replace('\n', '\n\t')
        text = '\n\n'.join([_main_help, title, intro, pyshell_doc])
        return self.parse_text(text)

    def parse_object(self, obj):
        """
        Parse the object's doc-string

        """
        if not hasattr(obj, '__doc__'):
            raise ValueError("Object does not have a doc-string.")

        attrs = {}
        # ## new customized parsing
        if isinstance(obj, (types.MethodType, types.BuiltinMethodType)):
            title = "%s(...)" % obj.__name__
            if isinstance(obj, types.MethodType):
                subtitle = "Method of %s" % str(type(obj.im_class))[1:-1]
            else:
                subtitle = "Method of %s" % str(type(obj.__self__))[1:-1]
            intro = doc2html(obj) + '<br>'
        elif isinstance(obj, (types.FunctionType, types.BuiltinFunctionType)):
            title = "%s(...)" % obj.__name__
            module = obj.__module__
            if module == '__builtin__':
                subtitle = "builtin function"
            else:
                subtitle = "function in %s" % module

            # function signature
            if inspect.isfunction(obj):
                a = inspect.getargspec(obj)
                if a.defaults:
                    n_args = len(a.args) - len(a.defaults)
                    args = a.args[:n_args]
                    args.extend(map('='.join, zip(a.args[n_args:], map(repr, a.defaults))))
                else:
                    args = a.args
                if a.varargs:
                    args.append('*%s' % a.varargs)
                if a.keywords:
                    args.append('**%s' % a.keywords)
                signature = "%s(%s)" % (obj.__name__, ', '.join(args))
                subtitle = '<br>'.join((subtitle, signature))

            intro = doc2html(obj)
        elif isinstance(obj, types.ObjectType):
            if isinstance(obj, (types.TypeType, types.ClassType)):
                # ClassType: old-style classes (?)
                title = "%s" % obj.__name__
                module = obj.__module__
            else:
                title = "%s" % obj.__class__.__name__
                module = obj.__class__.__module__
            if module == '__builtin__':
                subtitle = "builtin class"
            else:
                subtitle = "class in %s" % module
            intro = '<br>'.join((doc2html(obj),
                                 "<h2>Initialization</h2>",
#                                 format_subtitle("(...)"),
                                 doc2html(obj.__init__)))
            attrs_title = "Methods"
            for attr in dir(obj):
                if attr.startswith('_'):
                    continue
                a = getattr(obj, attr)
                if not hasattr(a, '__call__'):
                    continue
                attrs[attr] = "", doc2html(a)
        # ## OLD default parsing
        else:
            is_function = isinstance(obj, (types.FunctionType, types.BuiltinFunctionType))

            # doc-string for the object itself
            obj_type = str(type(obj))[1:-1]
            if hasattr(obj, '__name__'):
                title = obj.__name__
                subtitle = obj_type
            else:
                title = obj_type
                subtitle = None

            intro = doc2html(obj)
            if not is_function:
                intro += '<br><br><hr style="width: 100%; height: 2px;"><br><br>'

            # collect attributes
            if not is_function:
                attrs_title = "Attributes"
                for attr in dir(obj):
                    if attr[0] != '_' or attr == '__init__':
                        try:
                            a = getattr(obj, attr)
                        except:
                            pass
                        else:
                            typename = str(type(a))[6:-1]
                            attrs[attr] = typename, doc2html(a)

        # add text for attrs
        TOC = []
        chapters = []
        if len(attrs) > 0:
            for name in sorted(attrs):
                typename, doc = attrs[name]
                if doc:
                    TOC.append('<a href="#%s">%s</a> %s<br>' % (name, name, typename))
                    chapters.append('<h2><a href="#TOC">&uarr;</a><a name="%s"></a>%s</h2><br>%s' \
                                    % (name, name, doc))
                else:
                    TOC.append('<span style="color: rgb(102, 102, 102);">'
                               '%s (no __doc__)</span><br>' % name)

        # compose text
        txt = "<h1>%s</h1><br>" % title
        if subtitle:
            txt += '%s<br>' % format_subtitle(subtitle)
        txt += intro

        if len(TOC) > 0:
            txt += '<h1><a name="TOC"></a>%s</h1><br>' % attrs_title
            txt += ''.join(TOC)
            txt += '<br>'.join(chapters)

        if txt == '':
            txt = "Error: No doc-string found."

        return txt

    def parse_text(self, text):
        return rst2html(text)


_main_help = """
Eelbrain online documentation: https://pythonhosted.org/eelbrain/

"""

