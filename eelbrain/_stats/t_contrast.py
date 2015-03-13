# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from itertools import izip
import re

import numpy as np

from . import stats


# array functions:  work on array, take axis argument
np_afuncs = {'min': np.min,
             'max': np.max,
             'sum': np.sum}
# binary functions:  work on two arrays
np_bfuncs = {'subtract': np.subtract,
             'add': np.add}
# unary functions:  work on a single array
np_ufuncs = {'abs': np.abs,
             'negative': np.negative}


class TContrastRel(object):
    "Parse a contrast expression and expose methods to apply it"

    def __init__(self, contrast, cells, indexes):
        """Parse a contrast expression and expose methods to apply it

        Parameters
        ----------
        contrast : str
            Contrast specification.
        cells : tuple of cells
            Cells that occur in the contrast (each cell is represented by a str
            or a tuple of str).
        indexes : dict {cell: index}
            Indexes for the data of every cell.
        """
        parse = _parse_t_contrast(contrast)
        n_buffers, cells_in_contrast = _t_contrast_rel_properties(parse)
        pcells, mcells = _t_contrast_rel_expand_cells(cells_in_contrast, cells)

        self.contrast = contrast
        self.indexes = indexes
        self._parsed_contrast = parse
        self._pcells = pcells
        self._mcells = mcells
        self._n_buffers = n_buffers

        # data buffers
        self._buffer_shape = None
        self._buffer = None
        self._y_perm = None

    def map(self, y):
        "Apply contrast without retainig data buffers"
        buff = np.empty((self._n_buffers,) + y.shape[1:])
        data = _t_contrast_rel_data(y, self.indexes, self._pcells, self._mcells)
        tmap = _t_contrast_rel(self._parsed_contrast, data, buff)
        return tmap

    def __call__(self, y, out, perm):
        "Apply contrast to permutation of the data, storing and recycling data buffers"
        buffer_shape = (self._n_buffers,) + y.shape[1:]
        if self._buffer_shape != buffer_shape:
            self._buffer = np.empty(buffer_shape)
            self._y_perm = np.empty_like(y)
            self._buffer_shape = buffer_shape
        self._y_perm[perm] = y
        data = _t_contrast_rel_data(self._y_perm, self.indexes, self._pcells, self._mcells)
        tmap = _t_contrast_rel(self._parsed_contrast, data, self._buffer, out)
        return tmap


def _parse_cell(cell_name):
    "Parse a cell name for t_contrast"
    cell = tuple(s.strip() for s in cell_name.split('|'))
    if len(cell) == 1:
        return cell[0]
    else:
        return cell


def _parse_t_contrast(contrast):
    """Parse a string specifying a t-contrast into nested instruction tuples

    Parameters
    ----------
    contrast : str
        Contrast specification string.

    Returns
    -------
    compiled_contrast : tuple
        Nested tuple composed of:
        Comparisons:  ``('comp', c1, c0)`` and
        Unary functions:  ``('ufunc', func, arg)``
        Binary functions:  ``('bfunc', func, [arg1, arg2])``
        Array functions:  ``('afunc', func, [arg1, arg2, ...])``
        where ``arg1`` etc. are in turn comparisons and functions.
    """
    depth = 0
    start = 0
    if not '(' in contrast:
        m = re.match("\s*([\w\|*]+)\s*([<>])\s*([\w\|*]+)", contrast)
        if m:
            c1, direction, c0 = m.groups()
            if direction == '<':
                c1, c0 = c0, c1
            c1 = _parse_cell(c1)
            c0 = _parse_cell(c0)
            return ('comp', c1, c0)

    for i, c in enumerate(contrast):
        if c == '(':
            if depth == 0:
                prefix = contrast[start:i]
                i_open = i + 1
                items = []
            depth += 1
        elif c == ',':
            if depth == 0:
                raise
            elif depth == 1:
                item = _parse_t_contrast(contrast[i_open:i])
                items.append(item)
                i_open = i + 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                item = _parse_t_contrast(contrast[i_open:i])
                items.append(item)

                if contrast[i+1:].strip():
                    raise ValueError("Expression continues after last "
                                     "parentheses closed: %s" % contrast)
                elif prefix == '':
                    if len(items) == 1:
                        return items[0]
                    else:
                        raise ValueError("Multiple comparisons without "
                                         "combination expression: %s" % contrast)

                m = re.match("\s*(\w+)\s*", prefix)
                if m is None:
                    raise ValueError("uninterpretable prefix: %r" % prefix)
                func = m.group(1)
                if func in np_ufuncs:
                    if len(items) != 1:
                        raise ValueError("Wrong number of input values for "
                                         "unary function: %s" % contrast)
                    return 'ufunc', np_ufuncs[func], items[0]
                elif func in np_bfuncs:
                    if len(items) != 2:
                        raise ValueError("Wrong number of input values for "
                                         "binary function: %s" % contrast)
                    return 'bfunc', np_bfuncs[func], items
                elif func in np_afuncs:
                    if len(items) < 2:
                        raise ValueError("Wrong number of input values for "
                                         "array comparison function: %s"
                                         % contrast)
                    return 'afunc', np_afuncs[func], items
                else:
                    raise ValueError("Unknown function: %s" % contrast)
            elif depth == -1:
                err = "Invalid ')' at position %i of %r" % (i, contrast)
                raise ValueError(err)


def _t_contrast_rel_properties(item):
    """Find properties of a compiled t-contrast

    Parameters
    ----------
    item : tuple
        Contrast specification.

    Returns
    -------
    n_buffers : int
        Number of buffer maps needed.
    cells : set
        names of all cells that occur in the contrast.
    """
    if item[0] == 'ufunc':
        needed_buffers, cells = _t_contrast_rel_properties(item[2])
        return needed_buffers + 1, cells
    elif item[0] in ('bfunc', 'afunc'):
        _, _, items_ = item
        local_buffers = len(items_)
        cells = set()
        for i, item_ in enumerate(items_):
            available_buffers = local_buffers - i - 1
            needed_buffers, cells_ = _t_contrast_rel_properties(item_)
            additional_buffers = needed_buffers - available_buffers
            if additional_buffers > 0:
                local_buffers += additional_buffers
            cells.update(cells_)
        return local_buffers, cells
    else:
        return 0, set(item[1:])


def _t_contrast_rel_expand_cells(cells, all_cells):
    """Find cells that are an average of other cells

    Parameters
    ----------
    cells : set
        Cells occurring in the contrast.
    all_cells : tuple
        All cells in the data.

    Returns
    -------
    primary_cells : set
        All cells that occur directly in the data.
    mean_cells : dict
        ``{name: components}`` dictionary (components being a tuple with all
        cells to be averaged).
    """
    # check all cells have same number of components
    ns = set(1 if isinstance(cell, str) else len(cell) for cell in all_cells)
    ns.update(1 if isinstance(cell, str) else len(cell) for cell in cells)
    if len(ns) > 1:
        msg = ("Not all cells have the same number of components: %s" %
               str(tuple(cells) + tuple(all_cells)))
        raise ValueError(msg)

    primary_cells = set()
    mean_cells = {}
    for cell in cells:
        if cell in all_cells:
            primary_cells.add(cell)
        elif isinstance(cell, str):
            if cell != '*':
                raise ValueError("%s not in all_cells" % repr(cell))
            mean_cells[cell] = all_cells
            primary_cells.update(all_cells)
        elif not '*' in cell:
            msg = "Contrast contains cell not in data: %s" % repr(cell)
            raise ValueError(msg)
        else:
            # find cells that should be averaged ("base")
            base = tuple(cell_ for cell_ in all_cells if
                         all(i in (i_, '*') for i, i_ in izip(cell, cell_)))
            if len(base) == 0:
                raise ValueError("No cells in data match %s" % repr(cell))
            mean_cells[cell] = base
            primary_cells.update(base)

    return primary_cells, mean_cells


def _t_contrast_rel_data(y, indexes, cells, mean_cells):
    "Create {cell: data} dictionary"
    data = {}
    for cell in cells:
        index = indexes[cell]
        data[cell] = y[index]

    for name, cells_ in mean_cells.iteritems():
        cell = cells_[0]
        x = data[cell].copy()
        for cell in cells_[1:]:
            x += data[cell]
        x /= len(cells_)
        data[name] = x

    return data


def _t_contrast_rel(item, data, buff, out=None):
    "Execute a t_contrast (recursive)"
    if item[0] == 'ufunc':
        _, func, item_ = item
        tmap = _t_contrast_rel(item_, data, buff[1:], buff[0])
        tmap = func(tmap, tmap)
    elif item[0] == 'bfunc':
        _, func, items = item
        tmap1 = _t_contrast_rel(items[0], data, buff[2:], buff[1])
        tmap2 = _t_contrast_rel(items[1], data, buff[2:], buff[0])
        tmap = func(tmap1, tmap2, tmap2)
    elif item[0] == 'afunc':
        _, func, items_ = item
        tmaps = buff[:len(items_)]
        for i, item_ in enumerate(items_):
            _t_contrast_rel(item_, data, buff[i + 1:], tmaps[i])
        tmap = func(tmaps, axis=0, out=out)
    else:
        _, c1, c0 = item
        tmap = stats.t_1samp(data[c1] - data[c0], out)

    return tmap
