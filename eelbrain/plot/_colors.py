# -*- coding: utf-8 -*-
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Color tools for plotting."""
from __future__ import division

from itertools import izip, product
import operator

import numpy as np
import matplotlib as mpl

from .. import _colorspaces as cs
from .._data_obj import cellname, isfactor, isinteraction
from ._base import _EelFigure


def find_cell_colors(x, colors):
    """Process the colors arg from plotting functions

    Parameters
    ----------
    x : categorial
        Model for which colors are needed.
    colors : str | list | dict
        Colors for the plots if multiple categories of data are plotted.
        **str**: A colormap name; cells are mapped onto the colormap in
        regular intervals.
        **list**: A list of colors in the same sequence as cells.
        **dict**: A dictionary mapping each cell to a color.
        Colors are specified as `matplotlib compatible color arguments
        <http://matplotlib.org/api/colors_api.html>`_.
    """
    if isinstance(colors, (list, tuple)):
        cells = x.cells
        if len(colors) < len(cells):
            err = ("The `colors` argument %s does not supply enough "
                   "colors (%i) for %i "
                   "cells." % (str(colors), len(colors), len(cells)))
            raise ValueError(err)
        return dict(zip(cells, colors))
    elif isinstance(colors, dict):
        for cell in x.cells:
            if cell not in colors:
                raise KeyError("%s not in colors" % repr(cell))
        return colors
    elif colors is None or isinstance(colors, basestring):
        return colors_for_categorial(x, colors)
    else:
        raise TypeError("Invalid type: colors=%s" % repr(colors))


def colors_for_categorial(x, cmap=None):
    """Automatically select colors for a categorial model

    Parameters
    ----------
    x : categorial
        Model defining the cells for which to define colors.

    Returns
    -------
    colors : dict {cell -> color}
        Dictionary providing colors for the cells in x.
    """
    if isfactor(x):
        return colors_for_oneway(x.cells, cmap)
    elif isinteraction(x):
        return colors_for_nway([f.cells for f in x.base], cmap)
    else:
        msg = ("x needs to be Factor or Interaction, got %s" % repr(x))
        raise TypeError(msg)


def colors_for_oneway(cells, cmap='jet'):
    """Define colors for a single factor design

    Parameters
    ----------
    cells : sequence of str
        Cells for which to assign colors.
    cmap : str
        Name of a matplotlib colormap to use (default 'jet').

    Returns
    -------
    dict : {str: tuple}
        Mapping from cells to colors.
    """
    if cmap is None:
        cmap = 'jet'
    cm = mpl.cm.get_cmap(cmap)
    n = len(cells)
    return {cell: cm(i / n) for i, cell in enumerate(cells)}


def colors_for_twoway(x1_cells, x2_cells, cmap=None):
    """Define cell colors for a two-way design

    Parameters
    ----------
    x1_cells : tuple of str
        Cells of the major factor.
    x2_cells : tuple of str
        Cells of the minor factor.
    cmap : str
        Name of a matplotlib colormap to use (Default picks depending on number
        of cells in primary factor).

    Returns
    -------
    dict : {tuple: tuple}
        Mapping from cells to colors.
    """
    n1 = len(x1_cells)
    n2 = len(x2_cells)

    if n1 < 2 or n2 < 2:
        raise ValueError("Need at least 2 cells on each factor")

    if cmap is None:
        cm = cs.twoway_cmap(n1)
    else:
        cm = mpl.cm.get_cmap(cmap)

    # find locations in the color-space to sample
    n_colors = n1 * n2
    stop = (n_colors - 1) / n_colors
    samples = np.linspace(0, stop, n_colors)

    colors = dict(izip(product(x1_cells, x2_cells), map(tuple, cm(samples))))
    return colors


def colors_for_nway(cell_lists, cmap=None):
    """Define cell colors for a two-way design

    Parameters
    ----------
    cell_lists : sequence of of tuple of str
        List of the cells for each factor. E.g. for ``A % B``:
        ``[('a1', 'a2'), ('b1', 'b2', 'b3')]``.
    cmap : str
        Name of a matplotlib colormap to use (Default picks depending on number
        of cells in primary factor).

    Returns
    -------
    dict : {tuple: tuple}
        Mapping from cells to colors.
    """
    ns = map(len, cell_lists)

    if cmap is None:
        cm = cs.twoway_cmap(ns[0])
    else:
        cm = mpl.cm.get_cmap(cmap)

    # find locations in the color-space to sample
    n_colors = reduce(operator.mul, ns)
    edge = 0.5 / n_colors
    samples = np.linspace(edge, 1 - edge, n_colors)

    colors = {cell: tuple(color) for cell, color in
              izip(product(*cell_lists), cm(samples))}
    return colors


class ColorGrid(_EelFigure):
    """Plot colors for a two-way design in a grid

    Parameters
    ----------
    row_cells : tuple of str
        Cells contained in the rows.
    column_cells : tuple of str
        Cells contained in the columns.
    colors : dict
        Colors for cells.
    size : scalar
        Size (width and height) of the color squares (the default is to
        scale them to fit the figure).
    column_label_position : 'top' | 'bottom'
        Where to place the column labels (default is 'top').
    row_first : bool
        Whether the row cell precedes the column cell in color keys. By
        default this is inferred from the existing keys.
    """
    def __init__(self, row_cells, column_cells, colors, size=None,
                 column_label_position='top', row_first=None, *args, **kwargs):
        if row_first is None:
            row_cell_0 = row_cells[0]
            col_cell_0 = column_cells[0]
            if (row_cell_0, col_cell_0) in colors:
                row_first = True
            elif (col_cell_0, row_cell_0) in colors:
                row_first = False
            else:
                msg = ("Neither %s nor %s exist as a key in colors" %
                       ((row_cell_0, col_cell_0), (col_cell_0, row_cell_0)))
                raise KeyError(msg)

        if size is None:
            tight = True
        else:
            tight = False
        _EelFigure.__init__(self, "ColorGrid", None, 3, 1, tight, *args, **kwargs)
        ax = self.figure.add_axes((0, 0, 1, 1), frameon=False)
        ax.set_axis_off()
        self._ax = ax

        # reverse rows so we can plot upwards
        row_cells = tuple(reversed(row_cells))
        n_rows = len(row_cells)
        n_cols = len(column_cells)

        # color patches
        for col in xrange(n_cols):
            for row in xrange(n_rows):
                if row_first:
                    cell = (row_cells[row], column_cells[col])
                else:
                    cell = (column_cells[col], row_cells[row])
                patch = mpl.patches.Rectangle((col, row), 1, 1, fc=colors[cell],
                                              ec='none')
                ax.add_patch(patch)

        # column labels
        self._labels = []
        if column_label_position == 'top':
            y = n_rows + 0.1
            va = 'bottom'
            rotation = 40
            ymin = 0
            ymax = self._layout.h / size
        elif column_label_position == 'bottom':
            y = -0.1
            va = 'top'
            rotation = -40
            ymax = n_rows
            ymin = n_rows - self._layout.h / size
        else:
            msg = "column_label_position=%s" % repr(column_label_position)
            raise ValueError(msg)

        for col in xrange(n_cols):
            label = column_cells[col]
            h = ax.text(col + 0.5, y, label, va=va, ha='left', rotation=rotation)
            self._labels.append(h)

        # row labels
        x = n_cols + 0.1
        for row in xrange(n_rows):
            label = row_cells[row]
            h = ax.text(x, row + 0.5, label, va='center', ha='left')
            self._labels.append(h)

        if size is not None:
            self._ax.set_xlim(0, self._layout.w / size)
            self._ax.set_ylim(ymin, ymax)

        self._show()

    def _tight(self):
        # arbitrary default with equal aspect
        self._ax.set_ylim(0, 1)
        self._ax.set_xlim(0, 1 * self._layout.w / self._layout.h)

        # draw to compute text coordinates
        self.draw()

        # find label bounding box
        xmax = 0
        ymax = 0
        for h in self._labels:
            bbox = h.get_window_extent()
            if bbox.xmax > xmax:
                xmax = bbox.xmax
                xpos = h.get_position()[0]
            if bbox.ymax > ymax:
                ymax = bbox.ymax
                ypos = h.get_position()[1]
        xmax += 2
        ymax += 2

        # transform from display coordinates -> data coordinates
        trans = self._ax.transData.inverted()
        xmax, ymax = trans.transform((xmax, ymax))

        # calculate required movement
        _, ax_xmax = self._ax.get_xlim()
        _, ax_ymax = self._ax.get_ylim()
        xtrans = ax_xmax - xmax
        ytrans = ax_ymax - ymax

        # calculate the scale factor:
        # new_coord = x * coord
        # new_coord = coord + trans
        # x = (coord + trans) / coord
        scale = (xpos + xtrans) / xpos
        scale_y = (ypos + ytrans) / ypos
        if scale_y <= scale:
            scale = scale_y

        self._ax.set_xlim(0, ax_xmax / scale)
        self._ax.set_ylim(0, ax_ymax / scale)


class ColorList(_EelFigure):
    """Plot colors with labels

    Parameters
    ----------
    colors : dict
        Colors for cells.
    cells : tuple
        Cells for which to plot colors (default is ``colors.keys()``).
    h : 'auto' | scalar
        Height of the figure in inches. If 'auto' (default), the height is
        automatically increased to fit all labels.
    """
    def __init__(self, colors, cells=None, h='auto', *args, **kwargs):
        if h != 'auto':
            kwargs['h'] = h
        _EelFigure.__init__(self, "Colors", None, 2, 1.5, False, None, *args,
                            **kwargs)

        if cells is None:
            cells = colors.keys()

        ax = self.figure.add_axes((0, 0, 1, 1), frameon=False)
        ax.set_axis_off()

        n = len(cells)
        text_h = []
        for i, cell in enumerate(cells):
            bottom = n - i - 1
            y = bottom + 0.5
            patch = mpl.patches.Rectangle((0, bottom), 1, 1, fc=colors[cell],
                                          ec='none', zorder=1)
            ax.add_patch(patch)
            text_h.append(ax.text(1.1, y, cellname(cell), va='center', ha='left', zorder=2))

        ax.set_ylim(0, n)
        ax.set_xlim(0, n * self._layout.w / self._layout.h)

        # resize the figure to ft the content
        if h == 'auto':
            width, old_height = self._frame.GetSize()
            self.draw()
            text_height = max(h.get_window_extent().height for h in text_h) * 1.2
            new_height = text_height * n
            if new_height > old_height:
                self._frame.SetSize((width, new_height))

        self._show()


class ColorBar(_EelFigure):
    u"""A color-bar for a matplotlib color-map

    Parameters
    ----------
    cmap : str | Colormap
        Name of the color-map, or a matplotlib Colormap.
    vmin : scalar
        Lower end of the scale mapped onto cmap.
    vmax : scalar
        Upper end of the scale mapped onto cmap.
    label : None | str
        Label for the x-axis (default is the unit, or if no unit is provided
        the name of the colormap).
    label_position : 'left' | 'right' | 'top' | 'bottom'
        Position of the axis label. Valid values depend on orientation.
    label_rotation : scalar
        Angle of the label in degrees (For horizontal colorbars, the default is
        0; for vertical colorbars, the default is 0 for labels of 3 characters
        and shorter, and 90 for longer labels).
    clipmin : scalar
        Clip the color-bar below this value.
    clipmax : scalar
        Clip the color-bar above this value.
    orientation : 'horizontal' | 'vertical'
        Orientation of the bar (default is horizontal).
    unit : str
        Unit for the axis to determine tick labels (for example, ``u'µV'`` to
        label 0.000001 as '1').
    contours : iterator of scalar (optional)
        Plot contour lines at these values.
    """
    def __init__(self, cmap, vmin, vmax, label=True, label_position=None,
                 label_rotation=None,
                 clipmin=None, clipmax=None, orientation='horizontal',
                 unit=None, contours=(), *args, **kwargs):
        cm = mpl.cm.get_cmap(cmap)
        lut = cm(np.arange(cm.N))
        if orientation == 'horizontal':
            h = 1
            ax_aspect = 4
            im = lut.reshape((1, cm.N, 4))
        elif orientation == 'vertical':
            h = 4
            ax_aspect = 0.3
            im = lut.reshape((cm.N, 1, 4))
        else:
            raise ValueError("orientation=%s" % repr(orientation))

        if label is True:
            if unit:
                label = unit
            else:
                label = cm.name

        title = "ColorBar: %s" % cm.name
        _EelFigure.__init__(self, title, 1, h, ax_aspect, *args, **kwargs)
        ax = self._axes[0]

        if orientation == 'horizontal':
            ax.imshow(im, extent=(vmin, vmax, 0, 1), aspect='auto')
            ax.set_xlim(clipmin, clipmax)
            ax.yaxis.set_ticks(())
            self._contours = [ax.axvline(c, c='k') for c in contours]
            if unit:
                self._configure_xaxis(unit, label)
            elif label:
                ax.set_xlabel(label)

            if label_position is not None:
                ax.xaxis.set_label_position(label_position)

            if label_rotation is not None:
                ax.xaxis.label.set_rotation(label_rotation)
        elif orientation == 'vertical':
            ax.imshow(im, extent=(0, 1, vmin, vmax), aspect='auto', origin='lower')
            ax.set_ylim(clipmin, clipmax)
            ax.xaxis.set_ticks(())
            self._contours = [ax.axhline(c, c='k') for c in contours]
            if unit:
                self._configure_yaxis(unit, label)
            elif label:
                ax.set_ylabel(label)

            if label_position is not None:
                ax.yaxis.set_label_position(label_position)

            if label_rotation is not None:
                ax.yaxis.label.set_rotation(label_rotation)
                if (label_rotation + 10) % 360 < 20:
                    ax.yaxis.label.set_va('center')
            elif label and len(label) <= 3:
                ax.yaxis.label.set_rotation(0)
                ax.yaxis.label.set_va('center')
        else:
            raise ValueError("orientation=%s" % repr(orientation))

        self._show()
