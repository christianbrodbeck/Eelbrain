"""
Create text objects which can be output as str as well as tex.
The main class is :py:class:`~eelbrain.fmtxt.Table`, which can represent
tables in equal width font as well as tex.

:py:mod:`fmtxt` objects provide:

- a :py:meth:`__str__` method for a string representation
- a :py:meth:`get_gex` method for a tex code representation

The module also provides functions that work with fmtxt objects:

- :py:func:`save_tex` for saving an object's tex representation
- :py:func:`copy_tex` for copying an object's tex representation to
  the clipboard
- :py:func:`save_pdf` for saving a pdf
- :py:func:`copy_pdf` for copying a pdf to the clipboard



@author Christian M Brodbeck 2009; christianmbrodbeck@gmail.com
"""

import logging
import os
import tempfile

try:
    import tex
except:
    logging.warning("module tex not found; pdf export not available")
    tex = None

import numpy as np

from . import ui


preferences = dict(
                   keep_recent=3,  # number of recent tables to keep in memory
                   )


# to keep track of recent tex out and allow copying
_recent_texout = []
def _add_to_recent(tex_obj):
    keep_recent = preferences['keep_recent']
    if keep_recent:
        if len(_recent_texout) >= keep_recent - 1:
            _recent_texout.pop(0)
        _recent_texout.append(tex_obj)


def isstr(obj):
    return isinstance(obj, basestring)


def get_pdf(tex_obj):
    "creates a pdf from an fmtxt object (using tex)"
    txt = tex_obj.get_tex()
    document = u"""
\\documentclass{article}
\\usepackage{booktabs}
\\begin{document}
%s
\\end{document}
""" % txt
    pdf = tex.latex2pdf(document)
    return pdf


def save_pdf(tex_obj, path=None):
    "Save an fmtxt object as a pdf"
    pdf = get_pdf(tex_obj)
    if path is None:
        msg = "Save as PDF"
        path = ui.ask_saveas(msg, msg, [('PDF (*.pdf)', '*.pdf')])
    if path:
        with open(path, 'w') as f:
            f.write(pdf)


def save_tex(tex_obj, path=None):
    "saves an fmtxt object as a pdf"
    txt = tex_obj.get_tex()
    if path is None:
        path = ui.ask_saveas(title="Save tex", ext=[('tex', 'tex source code')])
    if path:
        with open(path, 'w') as f:
            f.write(txt)


def copy_pdf(tex_obj=-1):
    """
    copies an fmtxt object to the clipboard as pdf. `tex_obj` can be an object
    with a `.get_tex` method or an int, in which case the item is retrieved from
    a list of recently displayed fmtxt objects.

    """
    if isinstance(tex_obj, int):
        tex_obj = _recent_texout[tex_obj]

    # save pdf to temp file
    pdf = get_pdf(tex_obj)
    fd, path = tempfile.mkstemp('.pdf', text=True)
    os.write(fd, pdf)
    os.close(fd)
    logging.debug("Temporary file created at: %s" % path)

    # copy to clip-board
    ui.copy_file(path)


def copy_tex(tex_obj):
    "copies an fmtxt object to the clipboard as tex code"
    txt = tex_obj.get_tex()
    ui.copy_text(txt)


def texify(txt):
    """
    prepares non-latex txt for input to tex (e.g. for Matplotlib)

    """
    if hasattr(txt, 'get_tex'):
        return txt.get_tex()
    elif not isstr(txt):
        txt = str(txt)

    out = txt.replace('_', r'\_') \
             .replace('{', r'\{') \
             .replace('}', r'\}')
    return out


class texstr(object):
    """
    An object that stores a value along with formatting properties.

    The elementary unit of the :py:mod:`fmtxt` module. It can function as a
    string, but can hold formatting properties such as font properties.

    The following methods are used to get different string representations:

     - texstr.get_str() -> unicode
     - texstr.get_tex() -> str (tex)
     - texstr.__str__() -> str

    """
    def __init__(self, text, property=None, mat=False,
                 drop0=False, fmt='%.6g'):
        """
        An object that stores a value along with formatting properties.

        Parameters
        ----------
        text : object | list
            Any item with a string representation; can be a str, a texstr, a
            scalar, or a list containing a combination of those.
        property : str
            TeX property that is followed by {}
            (e.g., ``property=r'\textbf'`` will result in ``'\textbf{...}'``)
        mat : bool
            For TeX output, content is enclosed in ``'$...$'``
        drop0 : bool
            For  numbers smaller than 0, drop the '0' before the decimal
            point (e.g., for p values).
        fmt : str
            Format-str for numerical values.
        """
        if hasattr(text, 'get_tex'):  # texstr
            text = [text]
        elif (np.iterable(text)) and (not isstr(text)):  # lists
            t_new = []  # replace problematic stuff
            for t in text:
                if type(t) in [texstr, Stars]:
                    t_new.append(t)
                else:  # need to add texstr subclasses
                    t_new.append(texstr(t))
            text = t_new

        if isinstance(text, np.integer):
            # np integers are not recognized as instance of int
            self.text = int(text)
        else:
            self.text = text
        self.mat = mat
        self.drop0 = drop0
        self.fmt = fmt
        self.property = property

    def __add__(self, other):
        return texstr([self, other])

    def __repr_items__(self):
        items = [repr(self.text)]
        if self.property:
            items.append(repr(self.property))
        if self.mat:
            items.append('mat=True')
        if self.drop0:
            items.append('drop0=True')
        if self.fmt != '%s':
            items.append('fmt=%r' % self.fmt)
        return items

    def __repr__(self):
        name = self.__class__.__name__
        args = ', '.join(self.__repr_items__())
        return "%s(%s)" % (name, args)

    def __str__(self):
        return unicode(self).encode('utf-8')

    def __unicode__(self):
        return self.get_str()

    def get_str(self, fmt=None):
        """
        Returns the string representation.

        Parameters
        ----------
        fmt : str
            can be used to override the format string associated with the
            texstr object
        """
        if isstr(self.text):
            return self.text.replace('\\', '')
        elif np.iterable(self.text):
            return ''.join([str(t) for t in self.text])
        elif self.text is None:
            return ''
        elif np.isnan(self.text):
            return 'NaN'
        elif isinstance(self.text, (bool, np.bool_, np.bool8)):
            return '%s' % self.text
        elif np.isscalar(self.text) or getattr(self.text, 'ndim', None) == 0:
            if fmt:
                txt = fmt % self.text
            else:
                txt = self.fmt % self.text
            if self.drop0 and len(txt) > 2 and txt.startswith('0.'):
                txt = txt[1:]
            return txt
        elif not self.text:
            return ''
        else:
            msg = "Unknown text in tex.texstr: {0}".format(str(self.text))
            logging.warning(msg)
            return ''

    def get_tex(self, mat=False, fmt=None):
        if isstr(self.text) or not np.iterable(self.text):
            tex = self.get_str(fmt=fmt)
        else:
            tex = ''.join([tex_e.get_tex(mat=(self.mat or mat)) for tex_e in
                           self.text])

        if self.property:
            tex = r"%s{%s}" % (self.property, tex)
        if (self.mat) and (not mat):
            tex = "$%s$" % tex
        return tex


class symbol(texstr):
    "Print df neatly in plain text as well as formatted"
    def __init__(self, symbol, df=None):
        assert (df is None) or np.isscalar(df) or isstr(df) or np.iterable(df)
        self._df = df
        texstr.__init__(self, symbol)

    def get_df_str(self):
        if np.isscalar(self._df):
            return '%s' % self._df
        elif isstr(self._df):
            return self._df
        else:
            return ','.join(str(i) for i in self._df)

    def get_str(self, fmt=None):
        symbol = texstr.get_str(self, fmt=fmt)
        if self._df is None:
            return symbol
        else:
            return '%s(%s)' % (symbol, self.get_df_str())

    def get_tex(self, mat=False, fmt=None):
        out = texstr.get_tex(self, mat, fmt)
        if self._df is not None:
            out += '_{%s}' % self.get_df_str()

        if mat:
            return out
        else:
            return out.join(('$', '$'))


def p(p, digits=3, stars=None, of=3):
    """
    returns a texstr with properties set for p-values

    """
    if p < 10 ** -digits:  # APA 6th, p. 114
        p = '< .' + '0' * (digits - 1) + '1'
        mat = True
    else:
        mat = False
    fmt = '%' + '.%if' % digits
    ts_p = texstr(p, fmt=fmt, drop0=True, mat=mat)
    if stars is None:
        return ts_p
    else:
        ts_s = Stars(stars, of=of)
        return texstr((ts_p, ts_s), mat=True)


def stat(x, fmt="%.2f", stars=None, of=3, drop0=False):
    """
    returns a texstr with properties set for a statistic (e.g. a t-value)

    """
    ts_stat = texstr(x, fmt=fmt, drop0=drop0)
    if stars is None:
        return ts_stat
    else:
        ts_s = Stars(stars, of=of)
        return texstr((ts_stat, ts_s), mat=True)


def eq(name, result, eq='=', df=None, fmt='%.2f', drop0=False, stars=None,
       of=3):
    symbol_ = symbol(name, df=df)
    stat_ = stat(result, fmt=fmt, drop0=drop0, stars=stars, of=of)
    return texstr([symbol_, eq, stat_], mat=True)


def bold(txt):
    return texstr(txt, property=r'\textbf')


class Stars(texstr):
    """
    Shortcut for adding stars to a table and spaces in place of absent stars,
    so that alignment to the right can be used.

    n can be str but numbers are preferred.

    """
    def __init__(self, n, of=3, property="^"):
        if isstr(n):
            self.n = len(n.strip())
        else:
            self.n = n
        self.of = of
        if np.isreal(n):
            text = '*' * n + ' ' * (of - n)
        else:
            text = n.ljust(of)
        texstr.__init__(self, text, property=property)

    def get_tex(self, mat=False, fmt=None):
        txt = str(self)
        spaces = r'\ ' * (self.of - self.n)
        txtlist = ['^{', txt, spaces, '}']
        if not mat:
            txtlist = ['$'] + txtlist + ['$']
        return ''.join(txtlist)


# Table ---

class Cell(texstr):
    def __init__(self, text=None, property=None, width=1, just=None,
                 **texstr_kwargs):
        """A cell for a table

        Parameters
        ----------
        width : int
            Width in columns for multicolumn cells.
        just : None | 'l' | 'r' | 'c'
            Justification. None: use column standard.
        others :
            texstr parameters.
        """
        texstr.__init__(self, text, property, **texstr_kwargs)
        self.width = width
        if width > 1 and not just:
            self.just = 'l'
        else:
            self.just = just

    def __repr_items__(self):
        items = texstr.__repr_items__(self)
        if self.width != 1:
            i = min(2, len(items))
            items.insert(i, 'width=%s' % self.width)
        return items

    def __len__(self):
        return self.width

    def get_tex(self, fmt=None):
        tex_repr = texstr.get_tex(self, fmt=fmt)
        if self.width > 1 or self.just:
            tex_repr = r"\multicolumn{%s}{%s}{%s}" % (self.width, self.just,
                                                      tex_repr)
        return tex_repr


class Row(list):
    def __len__(self):
        return sum([len(cell) for cell in self])

    def __repr__(self):
        return "Row(%s)" % list.__repr__(self)

    def __str__(self):
        return unicode(self).encode('utf-8')

    def __unicode__(self):
        return ' '.join([str(cell) for cell in self])

    def _strlen(self, fmt=None):
        "returns list of cell-str-lengths; multicolumns handled poorly"
        lens = []
        for cell in self:
            cell_len = len(cell.get_str(fmt=fmt))
            for _ in xrange(len(cell)):
                lens.append(cell_len / len(cell))  # TODO: better handling of multicolumn
        return lens

    def get_str(self, c_width, c_just, delimiter='   ',
                fmt=None):
        "returns the row using col spacing provided in c_width"
        col = 0
        out = []
        for cell in self:
            if cell.width == 1:
                strlen = c_width[col]
                if cell.just:
                    just = cell.just
                else:
                    just = c_just[col]
            else:
                strlen = sum(c_width[col:col + cell.width])
                strlen += len(delimiter) * (cell.width - 1)
                just = cell.just
            col += cell.width
            txt = cell.get_str(fmt=fmt)
            if just == 'l':
                txt = txt.ljust(strlen)
            elif just == 'r':
                txt = txt.rjust(strlen)
            elif just == 'c':
                rj = strlen / 2
                txt = txt.rjust(rj).ljust(strlen)
            out.append(txt)
        return delimiter.join(out)

    def get_tex(self, fmt=None):
        tex = ' & '.join(cell.get_tex(fmt=fmt) for cell in self)
        tex += r" \\"
        return tex

    def get_tsv(self, delimiter, fmt=None):
        txt = delimiter.join(cell.get_str(fmt=fmt) for cell in self)
        return txt


class Table:
    """
    A table that can be output in text with equal width font
    as well as tex.

    Example::

        >>> from eelbrain import fmtxt
        >>> table = fmtxt.Table('lll')
        >>> table.cell()
        >>> table.cell("example 1")
        >>> table.cell("example 2")
        >>> table.midrule()
        >>> table.cell("string")
        >>> table.cell('???')
        >>> table.cell('another string')
        >>> table.cell("Number")
        >>> table.cell(4.5)
        >>> table.cell(2./3, fmt='%.4g')
        >>> print table
                 example 1   example 2
        -----------------------------------
        string   ???         another string
        Number   4.5         0.6667
        >>> print table.get_tex()
        \begin{center}
        \begin{tabular}{lll}
        \toprule
         & example 1 & example 2 \\
        \midrule
        string & ??? & another string \\
        Number & 4.5 & 0.6667 \\
        \bottomrule
        \end{tabular}
        \end{center}
        >>> table.save_tex()


    """
    def __init__(self, columns, rules=True, title=None, caption=None, rows=[]):
        """
        Parameters
        ----------
        columns : str
            alignment for each column, e.g. ``'lrr'``
        rules : bool
            Add toprule and bottomrule
        title : None | text
            Title for the table.
        caption : None | text
            Caption for the table.
        rows : list of Row
            Table body.
        """
        self.columns = columns
        self._table = rows[:]
        self.rules = rules
        self.title(title)
        self.caption(caption)
        self._active_row = None

    @property
    def shape(self):
        return (len(self._table), len(self.columns))

    def __len__(self):
        return len(self._table)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._table[item]
        else:
            rows = self._table[item]
            return Table(self.columns, rules=self.rules, title=self._title,
                         caption=self._caption, rows=rows)

    # adding texstrs ---
    def cell(self, *args, **kwargs):
        """
        args:   text, *properties
        OR:     texstr object

        properties are tex text properties (e.g. "textbf")


        kwargs
        ------

        Cell kwargs, e.g.:
        width=1     use value >1 for multicolumn cells
        just='l'    justification (only for multicolumn)
        mat=False   enclose tex output in $...$ if True
        texstr kwargs ...


        number properties
        -----------------

        drop0=False     drop 0 before
        digits=4        number of digits after dot


        Properties Example
        ------------------
        >>> table.cell("Entry", "textsf", "textbf") for bold sans serif
        """
        if len(args) == 0:
            txt = ''
        else:
            txt = args[0]

        if len(args) > 1:
            property = args[1]
        else:
            property = None

        cell = Cell(text=txt, property=property, **kwargs)

        if self._active_row is None or len(self._active_row) == len(self.columns):
            new_row = Row()
            self._table.append(new_row)
            self._active_row = new_row

        if len(cell) + len(self._active_row) > len(self.columns):
            raise ValueError("Cell too long -- row width exceeds table width")
        self._active_row.append(cell)

    def empty_row(self):
        self.endline()
        self._table.append(Row())

    def endline(self):
        "finishes the active row"
        if self._active_row is not None:
            for _ in xrange(len(self.columns) - len(self._active_row)):
                self._active_row.append(Cell())
        self._active_row = None

    def cells(self, *cells):
        "add several simple cells with one command"
        for cell in cells:
            self.cell(cell)

    def midrule(self, span=None):
        """
        adds midrule; span ('2-4' or (2, 4)) specifies the extent

        note that a toprule and a bottomrule are inserted automatically
        in every table.
        """
        self.endline()
        if span is None:
            self._table.append("\\midrule")
        else:
            if type(span) in [list, tuple]:
                # TODO: assert end is not too big
                span = '-'.join([str(int(i)) for i in span])
            assert '-' in span
            assert all([i.isdigit() for i in span.split('-')])
            self._table.append(r"\cmidrule{%s}" % span)

    def title(self, *args, **kwargs):
        """Set the table title (with texstr args/kwargs)"""
        if (len(args) == 1) and (args[0] is None):
            self._title = None
        else:
            self._title = texstr(*args, **kwargs)

    def caption(self, *args, **kwargs):
        """Set the table caption (with texstr args/kwargs)"""
        if (len(args) == 1) and (args[0] is None):
            self._caption = None
        else:
            self._caption = texstr(*args, **kwargs)

    def __repr__(self):
        """
        return self.__str__ so that when a function returns a Table, the
        result can be inspected without assigning the Table to a variable.

        """
        return self.__str__()

    def __str__(self):
        return unicode(self).encode('utf-8')

    def __unicode__(self):
        return self.get_str()

    def get_str(self, fmt=None, delim='   ', linesep=os.linesep):
        """Convert Table to str

        Parameters
        ----------
        fmt : None  | str
            Format for numbers.
        delim : str
            Delimiter between columns.
        linesep : str
            Line separation string
        """
        # append to recent tex out
        _add_to_recent(self)

        # determine column widths
        widths = []
        for row in self._table:
            if not isstr(row):  # some commands are str
                row_strlen = row._strlen()
                while len(row_strlen) < len(self.columns):
                    row_strlen.append(0)
                widths.append(row_strlen)
        try:
            widths = np.array(widths)
        except Exception, exc:
            print widths
            raise Exception(exc)
        c_width = np.max(widths, axis=0)  # column widths!

        # FIXME: take into account tab length:
        midrule = delim.join(['-' * w for w in c_width])
        midrule = midrule.expandtabs(4).replace(' ', '-')

        # collect lines
        txtlines = []
        for row in self._table:
            if isstr(row):  # commands
                if row == r'\midrule':
                    txtlines.append(midrule)  # "_"*l_len)
                elif row == r'\bottomrule':
                    txtlines.append(midrule)  # "_"*l_len)
                elif row == r'\toprule':
                    txtlines.append(midrule)  # "_"*l_len)
                elif row.startswith(r'\cmidrule'):
                    txt = row.split('{')[1]
                    txt = txt.split('}')[0]
                    start, end = txt.split('-')
                    start = int(start) - 1
                    end = int(end)
                    line = [' ' * w for w in c_width[:start]]
                    rule = delim.join(['-' * w for w in c_width[start:end]])
                    rule = rule.expandtabs(4).replace(' ', '-')
                    line += [rule]
                    line += [' ' * w for w in c_width[start:end]]
                    txtlines.append(delim.join(line))
                else:
                    pass
            else:
                txtlines.append(row.get_str(c_width, self.columns, fmt=fmt,
                                            delimiter=delim))
        out = txtlines

        if self._title != None:
            out = ['', self._title.get_str(), ''] + out

        if isstr(self._caption):
            out.append(self._caption)
        elif self._caption:
            out.append(str(self._caption))

        return linesep.join(out)

    def get_tex(self, fmt=None):
        tex_pre = [r"\begin{center}",
                   r"\begin{tabular}{%s}" % self.columns]
        if self.rules:
            tex_pre.append(r"\toprule")
        # Body
        tex_body = []
        for row in self._table:
            if isstr(row):
                tex_body.append(row)
            else:
                tex_body.append(row.get_tex(fmt=fmt))
        # post
        tex_post = [r"\end{tabular}",
                    r"\end{center}"]
        if self.rules:
            tex_post = [r"\bottomrule"] + tex_post
        # combine
        tex_repr = os.linesep.join(tex_pre + tex_body + tex_post)
        return tex_repr

    def get_tsv(self, delimiter='\t', linesep='\r\n', fmt='%.9g'):
        """
        Returns the table as tsv string.

        kwargs
        ------
        delimiter: delimiter between columns (by default tab)
        linesep:   delimiter string between lines
        fmt:       format for numerical entries
        """
        table = []
        for row in self._table:
            if isstr(row):
                pass
            else:
                table.append(row.get_tsv(delimiter, fmt=fmt))
        return linesep.join(table)

    def copy_pdf(self):
        "copy pdf to clipboard"
        copy_pdf(self)

    def copy_tex(self):
        "copy tex t clipboard"
        copy_tex(self)

    def save_pdf(self, path=None):
        "saves table on pdf; if path == non ask with system dialog"
        save_pdf(self, path=path)

    def save_tex(self, path=None):
        "saves table as tex; if path == non ask with system dialog"
        save_tex(self, path=path)

    def save_tsv(self, path=None, delimiter='\t', linesep='\r\n', fmt='%.15g'):
        """
        Save the table as tab-separated values file.

        Parameters
        ----------
        path : str | None
            Destination file name.
        delimiter : str
            String that is placed between cells (default: tab).
        linesep : str
            String that is placed in between lines.
        fmt : str
            Format string for representing numerical cells.
            (see 'Python String Formatting Documentation
            <http://docs.python.org/library/stdtypes.html#string-formatting-operations>'_ )
        """
        if not path:
            path = ui.ask_saveas(title="Save Tab Separated Table",
                                 message="Please Pick a File Name",
                                 ext=[("txt", "txt (tsv) file")])
        if ui.test_targetpath(path):
            ext = os.path.splitext(path)[1]
            if ext == '':
                path += '.txt'

            with open(path, 'w') as f:
                out = self.get_tsv(delimiter=delimiter, linesep=linesep,
                                   fmt=fmt)
                if isinstance(out, unicode):
                    out = out.encode('utf-8')
                f.write(out)

    def save_txt(self, path=None, fmt='%.15g', delim='   ', linesep=os.linesep):
        """
        Save the table as text file.

        Parameters
        ----------
        path : str | None
            Destination file name.
        fmt : str
            Format string for representing numerical cells.
        linesep : str
            String that is placed in between lines.
        """
        if not path:
            path = ui.ask_saveas(title="Save Table as Text File",
                                 message="Please Pick a File Name",
                                 ext=[("txt", "txt file")])
        if ui.test_targetpath(path):
            ext = os.path.splitext(path)[1]
            if ext == '':
                path += '.txt'

            with open(path, 'w') as f:
                out = self.get_str(fmt, delim, linesep)
                if isinstance(out, unicode):
                    out = out.encode('utf-8')
                f.write(out)


def unindent(text, skip1=False):
    """
    removes the minimum number of leading spaces present in all lines.

    skip1 : bool
        Ignore the first line when determining number of spaces to unindent,
        and remove all leading whitespaces from it.

    """
    # count leading whitespaces
    lines = text.splitlines()
    ws_lead = []
    for line in lines[skip1:]:
        len_stripped = len(line.lstrip(' '))
        if len_stripped:
            ws_lead.append(len(line) - len_stripped)

    if len(ws_lead) > skip1:
        rm = min(ws_lead)
        if rm:
            if skip1:
                lines[0] = ' ' * rm + lines[0].lstrip()

            text = os.linesep.join(line[rm:] for line in lines)

    return text
