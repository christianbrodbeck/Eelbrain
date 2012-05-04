"""
Table class 
-----------
Create tables which can be output in equal width font as well as tex.

wx is used for selecting a filename when saving as tex and can be deactivated 
if this function is not needed.

Usage:
table = tex.Table('lll')
table.cell()
table.cell("Hamster")
table.cell("Ethel")
table.midrule()
table.cell("Weight")
table.cell(2./3, digits=2)
table.cell(5./2, digits=2)
table.cell("Height")
table.cell("<1 foot")
table.cell("4 foot 3", "textbf")

print table
table.savetex()



Text objects need 
 - __str__    -> string for terminal
 - get_tex    -> tex code



(Written by Christian Brodbeck 2009; ChristianMBrodbeck@gmail.com)
"""

import os
import logging
import tempfile

try:
    import tex
except:
    logging.warning("module tex not found; tex functions not available")
    tex = None

import numpy as np

#try:
#    from eelbrain.ui import wx as ui
#except ImportError:
#    logging.warning("wx unavailable; using shell ui")
#    from eelbrain.ui import terminal as ui
from eelbrain import ui


defaults = dict(table_del = '   ') # ' \t'



# to keep track of recent tex out and allow copying
_recent_texout = []
_recent_len = 3
def _add_to_recent(tex_obj):
    if len(_recent_texout) >= _recent_len-1:
        _recent_texout.pop(0)
    _recent_texout.append(tex_obj)



def isstr(obj):
    return isinstance(obj, basestring)

def get_pdf(tex_obj):
    "creates a pdf from a textab object (using tex)"
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
    "saves a textab object as a pdf"
    pdf = get_pdf(tex_obj)
    if path is None:
        path = ui.ask_saveas(title="Save tex as pdf", ext=[('pdf','pdf')])
    if path:
        with open(path, 'w') as f:
            f.write(pdf)

def save_tex(tex_obj, path=None):
    "saves a textab object as a pdf"
    txt = tex_obj.get_tex()
    if path is None:
        path = ui.ask_saveas(title="Save tex", ext=[('tex','tex source code')])
    if path:
        with open(path, 'w') as f:
            f.write(txt)

def copy_pdf(tex_obj=-1):
    """
    copies a textab object to the clipboard as pdf. `tex_obj` can be an object 
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
    logging.debug("Temporary file created at: %s"%path)
    
    # copy to clip-board
    ui.copy_file(path)


def copy_tex(tex_obj):
    "copies a textab object to the clipboard as tex code"
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
    item that can function as a string, but can hold a more complete 
    representation (e.g. font properties) at the same time.
    
    The following methods are used to get different string representations:
     - texstr.__str__()
     - texstr.get_tex()
     
    """
    def __init__(self, text, property=None, mat=False,
                 drop0=False, fmt='%.6g'):
        """
        :arg text: can be a string, a texstr, a scalar, or a list containing 
            a combination of those.
        
        :arg str property: tex property that is followed by {} 
            (e.g., r'\textbf' will result in '\textbf{...}')
        
        :arg bool mat: for tex output, content is enclosed in $...$
        
        
        number properties
        -----------------
        
        fmt='%.6g': changes the standard representation (only affects 
            numerical values). Can be overwritten ...
        
        drop0=False: drop 0 before the point (for p values)

        """
#        logging.debug("tex.texstr.__init__(%s)"%str(text))
        if hasattr(text, 'get_tex'): #texstr
            text = [text]
        elif (np.iterable(text)) and (not isstr(text)): #lists
            t_new = [] # replace problematic stuff
            for t in text:
                if type(t) in [texstr, Stars]: 
                    t_new.append(t)
                else: # need to add texstr subclasses
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
        items = []
        if isstr(self.text):
            items.append('"%s"'%self.text)
        else:
            items.append(self.text.__repr__())
        
        if self.property:
            items.append(repr(self.property))
        if self.mat:
            items.append('mat=True')
        if self.drop0:
            items.append('drop0=True')
        if self.fmt != '%.6g':
            items.append('fmt=%r'%self.fmt)
        return items
    
    def __repr__(self):
        name = self.__class__.__name__
        args = ', '.join(self.__repr_items__())
        return "%s(%s)"%(name, args)
    
    def __str__(self):
        return self.get_str()
    
    def get_str(self, fmt=None):
        """
        Returns the string representation. 
        
        :kwarg str fmt: can be used to override the format string associated
            with the texstr object
        
        """
        if isstr(self.text):
            return self.text.replace('\\','')
        elif np.iterable(self.text):
            return ''.join([str(t) for t in self.text])
        elif self.text is None:
            return ''
        elif np.isnan(self.text):
            return 'NaN'
        elif isinstance(self.text, bool):
            return '%s' % self.text
        elif np.isscalar(self.text):
            if int(self.text) == self.text:
                return str(int(self.text))
            else:
                if fmt: 
                    txt = fmt % self.text
                else:
                    txt = self.fmt % self.text
                if self.drop0 and len(txt)>0 and txt[0]=='0':
                    txt = txt[1:]
                return txt
        elif not self.text:
            return ''
        else:
            logging.warning(" Unknown text in tex.texstr: {0}".format(str(self.text)))
            return ''
    
    def get_tex(self, mat=False, fmt=None):
        #print type(self.text)
        if isstr(self.text) or not np.iterable(self.text):
            tex = texstr.get_str(self, fmt=fmt)
        else:
            tex = ''.join([tex_e.get_tex(mat=(self.mat or mat)) for tex_e in self.text])
        if self.property:
            tex = r"%s{%s}"%(self.property, tex)
        if (self.mat) and (not mat):
            tex = "$%s$"%tex
        return tex


class symbol(texstr):
    def __init__(self, symbol, df=None):
        assert np.isscalar(df) or isstr(df) or np.iterable(df)
        self._df = df
        texstr.__init__(self, symbol)
    
    def get_str(self, fmt=None):
        symbol = texstr.get_str(self, fmt=fmt)
        return '%s(%s)'%(symbol, self.get_df_str())
    
    def get_df_str(self):
        if np.isscalar(self._df):
            return '%s' % self._df
        elif isstr(self._df):
            return self._df
        else:
            return ','.join(str(i) for i in self._df)
    
    def get_tex(self, mat=False, fmt=None):
        out = texstr.get_tex(self, mat, fmt)
        out += '_{%s}' % self.get_df_str()
        if mat:
            return out
        else:
            return out.join(('$', '$'))


## convenience texstr generators ######   ######   ######   ######   ######
def p(p, digits=3, stars=None, of=3):
    """
    returns a texstr with properties set for p-values
    
    """
    if p < 10**-digits:
        p = '< .' + '0'*(digits-1) + '1'
    fmt = '%' + '.%if'%digits
    ts_p = texstr(p, fmt=fmt, drop0=True)
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


def eq(name, result, eq='=', df=None, fmt='%.2f', drop0=False, 
       stars=None, of=3):
    return texstr([symbol(name, df=df), 
                   eq,
                   stat(result, fmt=fmt, drop0=drop0, stars=stars, of=of)],
                  mat=True)


def bold(txt):
    return texstr(txt, property = r'\textbf')


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
            text = '*'*n + ' '*(of-n)
        else:
            text = n.ljust(of)
        texstr.__init__(self, text, property=property)
     
    def get_tex(self, mat=False, fmt=None):
        txt = self.__str__()
        spaces = r'\ ' * (self.of - self.n)
        txtlist = ['^{', txt, spaces, '}']
        if not mat:
            txtlist = ['$'] + txtlist + ['$']
        return ''.join(txtlist)
        
        
##############################################################################

class Cell(texstr):
    def __init__(self, text=None, property=None, width=1, just=False, **texstr_kwargs):
        """
        width, pos: for multicolumn
        just: False = use column standard
              'l' / 'r' / ...
        
        properties: list of Tex Text Property commands, 
        e.g. Cell("Entry", "textsf", "textbf") for bold sans serif
        
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
            items.insert(i, 'width=%s'%self.width)
        return items
    
    def __len__(self):
        return self.width
    
    def get_tex(self, fmt=None):
        tex = texstr.get_tex(self, fmt=fmt)
        if self.width > 1 or self.just:
            tex = r"\multicolumn{%s}{%s}{%s}"%(self.width, self.just, tex)
        return tex


class Row(list):
    def __len__(self):
        return sum([len(cell) for cell in self])
    
    def get_tex(self, fmt=None):
        tex = ' & '.join(cell.get_tex(fmt=fmt) for cell in self)
        tex += r" \\"
        return tex
    
    def get_tsv(self, delimiter, fmt=None):
        txt = delimiter.join(cell.get_str(fmt=fmt) for cell in self)
        return txt
    
    def _strlen(self, fmt=None):
        "returns list of cell-str-lengths; multicolumns handled poorly"
        lens = []
        for cell in self:
            cell_len = len(cell.get_str(fmt=fmt))
            for i in xrange(len(cell)):
                lens.append(cell_len / len(cell)) # TODO: better handling of multicolumn
        return lens
    
    def __repr__(self):
        return "Row(%s)" % list.__repr__(self)
    
    def __str__(self):
        return ' '.join([str(cell) for cell in self])
    
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
                strlen = sum(c_width[col:col+cell.width])
                strlen += len(delimiter) * (cell.width-1)
                just = cell.just
            col += cell.width
            txt = cell.get_str(fmt=fmt)
            if just == 'l':
                txt = txt.ljust(strlen)
            elif just == 'r':
                txt = txt.rjust(strlen)
            elif just == 'c':
                rj = strlen/2
                txt = txt.rjust(rj).ljust(strlen)
            out.append(txt)
#        logging.debug("TEX: %s"%str([len(e) for e in out]))
        return delimiter.join(out)




class Table:
    """
    creates and stores a table that can be output in text with equal width font
    as well as tex
    
    Example::
    
    >>> table = tex.Table('lll')
    >>> table.cell()    # empty cell top left
    >>> table.cell("Outside", "textbf")
    >>> table.cell("Inside", "textbf")
    >>> table.midrule()
    >>> table.cell("Duck")
    >>> table.cell("Feathers")
    >>> table.cell("Duck Meat")
    >>> table.cell("Dog")
    >>> table.cell("Fur")
    >>> table.cell("Hotdog")
    >>> print table
    >>> print table.tex()
    >>> table.savetex()
    
    """
    def __init__(self, columns, rules=True, title=None, caption=None, rows=[]):
        """
        columns e.g. 'lrr'
        
        """
        self.columns = columns
        self._table = rows[:]
        self.rules = rules      # whether to add top and bottom rules
        self.title(title)
        self.caption(caption)
        self._active_row = None
    
    @property
    def shape(self):
        return (len(self._table), len(self.columns))
    
    def __len__(self):
        return len(self._table)
    
    def __getitem__(self, item):
        rows = self._table[item]
        return Table(self.columns, rules=self.rules, title=self._title,
                     caption=self._caption, rows=rows)
    
    """
    adding texstrs
    --------------
    
    """
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
    
    def endline(self):
        "finishes the active row"
        if self._active_row is not None:
            for i in xrange(len(self.columns) - len(self._active_row)):
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
            self._table.append(r"\cmidrule{%s}"%span)
    
    def title(self, *args, **kwargs): # TODO: title / caption
        if (len(args) == 1) and (args[0] is None):
            self._title = None
        else:
            self._title = texstr(*args, **kwargs)
    
    def caption(self, *args, **kwargs):
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
        return self.get_str()
    
    def get_str(self, fmt=None, row_delimiter='default'):
        # append to recent tex out
        _add_to_recent(self)
        
        if row_delimiter == 'default':
            row_delimiter = defaults['table_del']
        
        # determine column widths
        widths = []
        for row in self._table:
            if not isstr(row): # some commands are str
                row_strlen = row._strlen()
                while len(row_strlen) < len(self.columns):
                    row_strlen.append(0)
                widths.append(row_strlen)
        try:
            widths = np.array(widths)
        except Exception, exc:
            print widths
            raise Exception(exc)
        c_width = widths.max(axis=0) # column widths!
        
        # FIXME: take into account tab length:
        midrule = row_delimiter.join(['-'*w for w in c_width])
        midrule = midrule.expandtabs(4).replace(' ','-')
        
        # collect lines
        txtlines = []
        for row in self._table:
            if isstr(row): # commands
                if row == r'\midrule':
                    txtlines.append(midrule) #"_"*l_len)
                elif row == r'\bottomrule':
                    txtlines.append(midrule) #"_"*l_len)
                elif row == r'\toprule':
                    txtlines.append(midrule) #"_"*l_len)
                elif row.startswith(r'\cmidrule'):
                    txt = row.split('{')[1]
                    txt = txt.split('}')[0]
                    start, end = txt.split('-')
                    start = int(start) - 1
                    end = int(end)
                    line = [' '*w for w in c_width[:start]]
                    rule = row_delimiter.join(['-'*w for w in c_width[start:end]])
                    rule = rule.expandtabs(4).replace(' ','-')
                    line += [rule]
                    line +=[' '*w for w in c_width[start:end]]
                    txtlines.append(row_delimiter.join(line))
                else:
                    pass
            else:
                txtlines.append(row.get_str(c_width, self.columns, fmt=fmt, 
                                            delimiter=row_delimiter))
        out = txtlines
        
        if self._title != None:
            out = ['', self._title.get_str(), ''] + out
        
        if self._caption:
            out.append(str(self._caption))
        
        return '\n'.join(out)
    
    def get_tex(self, fmt=None):
        tex_pre = [r"\begin{center}", 
                   r"\begin{tabular}{%s}"%self.columns]
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
        tex = '\n'.join(tex_pre + tex_body + tex_post)
        return tex
    
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
        Saves the table as tab-separated values file. 
        
        :arg str delimiter: string that is placed between cells (default: tab).
        :arg str linesep: string that is placed in between lines.
        :arg str fmt: format string for representing numerical cells. 
            (see 'Python String Formatting Documentation <http://docs.python.org/library/stdtypes.html#string-formatting-operations>'_ )
            http://docs.python.org/library/stdtypes.html#string-formatting-operations
            
        """
        if not path:
            path = ui.ask_saveas(title = "Save Tab Separated Table",
                                 message = "Please Pick a File Name",
                                 ext = [("txt", "txt (tsv) file")])
        if ui.test_targetpath(path):
            ext = os.path.splitext(path)[1]
            if ext not in ['.tsv', '.txt']:
                path += '.txt'
            with open(path, 'w') as f:
                f.write(self.get_tsv(delimiter=delimiter, linesep=linesep, 
                                     fmt=fmt))
