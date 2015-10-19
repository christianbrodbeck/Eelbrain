# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Help Viewer"""

import inspect
import logging
import types

import wx
from wx.html2 import WebView
import docutils.core

from .. import fmtxt
from ..fmtxt import _html_doc_template
from .frame import EelbrainFrame


def show_help_rst(text, parent):
    html = rst2html(text)
    frame = HelpFrame(parent)
    frame.SetPage(html, "blah")
    frame.Show()


def show_help_txt(text, parent, title=""):
    "Show help frame with text in mono-spaced font"
    lines = (line.replace(' ', '&nbsp;') for line in text.splitlines())
    body = "<code>%s</code>" % '<br>'.join(lines)
    html = _html_doc_template.format(title="Help: %s" % title, body=body)
    frame = HelpFrame(parent)
    frame.SetPage(html, title)
    frame.Show()


def show_help_obj(obj):
    frame = HelpFrame(None)
    text = format_help(obj)
    frame.SetPage(text, str(obj))
    frame.Show()


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
        html = '<pre class="literal-block">%s</pre>' % rst
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


def format_subtitle(subtitle):
    return '<span style="color: rgb(102, 102, 102);">%s</span>' % subtitle


def format_help(obj):
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
                             doc2html(obj.__init__)))
        attrs_title = "Methods"
        for attr in dir(obj):
            if attr.startswith('_'):
                continue
            a = getattr(obj, attr)
            if not hasattr(a, '__call__'):
                continue
            attrs[attr] = "", doc2html(a)
    else:
        # doc-string for the object itself
        obj_type = str(type(obj))[1:-1]
        if hasattr(obj, '__name__'):
            title = obj.__name__
            subtitle = obj_type
        else:
            title = obj_type
            subtitle = None

        intro = doc2html(obj)
        intro += '<br><br><hr style="width: 100%; height: 2px;"><br><br>'

        # collect attributes
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


class HelpFrame(EelbrainFrame):
    #  http://stackoverflow.com/a/10866495/166700
    def __init__(self, parent, *args, **kwargs):
        display_w, display_h = wx.DisplaySize()
        x = 0
        y = 25
        w = min(650, display_w)
        h = min(1000, display_h - y)
        EelbrainFrame.__init__(self, parent, pos=(x, y), size=(w, h), *args, **kwargs)
        self.webview = WebView.New(self)

    def SetPage(self, html, url):
        self.webview.SetPage(html, url)
        self.SetTitle(self.webview.GetCurrentTitle())
