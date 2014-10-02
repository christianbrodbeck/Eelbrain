# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Color tools for plotting."""
from __future__ import division

from itertools import product

import numpy as np
import matplotlib as mpl

from .._data_obj import cellname
from ._base import _EelFigure


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
        if n1 == 2:
            cmap = "2group-ob"
        else:
            raise NotImplementedError("Major factor with more than 2 cells")
    cm = mpl.cm.get_cmap(cmap)

    # find locations in the color-space to sample
    n_colors = n1 * n2
    stop = (n_colors - 1) / n_colors
    samples = np.linspace(0, stop, n_colors)

    cs = map(tuple, cm(samples))
    colors = {cell: cs[i] for i, cell in enumerate(product(x1_cells, x2_cells))}
    return colors


class ColorGrid(_EelFigure):

    def __init__(self, row_cells, col_cells, colors, row_first=None,
                 **layout):
        """Plot colors in a grid

        Parameters
        ----------
        row_cells : tuple of str
            Cells contained in the rows.
        col_cells : tuple of str
            Cells contained in the columns.
        colors : dict
            Colors for cells.
        row_first : bool
            Whether the row cell precedes the column cell in color keys. By
            default this is inferred from the existing keys.
        """
        if row_first is None:
            row_cell_0 = row_cells[0]
            col_cell_0 = col_cells[0]
            if (row_cell_0, col_cell_0) in colors:
                row_first = True
            elif (col_cell_0, row_cell_0) in colors:
                row_first = False
            else:
                msg = ("Neither %s nor %s exist as a key in colors" %
                       ((row_cell_0, col_cell_0), (col_cell_0, row_cell_0)))
                raise KeyError(msg)

        _EelFigure.__init__(self, "ColorGrid", None, 3, 1, layout, True)
        ax = self.figure.add_axes((0, 0, 1, 1), frameon=False)
        ax.set_axis_off()
        self._ax = ax

        # reverse rows so we can plot upwards
        row_cells = tuple(reversed(row_cells))
        n_rows = len(row_cells)
        n_cols = len(col_cells)

        # color patches
        for col in xrange(n_cols):
            for row in xrange(n_rows):
                if row_first:
                    cell = (row_cells[row], col_cells[col])
                else:
                    cell = (col_cells[col], row_cells[row])
                patch = mpl.patches.Rectangle((col, row), 1, 1, fc=colors[cell],
                                              ec='none')
                ax.add_patch(patch)

        # labels
        self._labels = []
        y = n_rows + 0.1
        for col in xrange(n_cols):
            label = col_cells[col]
            h = ax.text(col + 0.5, y, label, va='bottom', ha='left',
                        rotation=40)
            self._labels.append(h)
        x = n_cols + 0.1
        for row in xrange(n_rows):
            label = row_cells[row]
            h = ax.text(x, row + 0.5, label, va='center', ha='left')
            self._labels.append(h)

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


class Colors(_EelFigure):

    def __init__(self, colors, cells=None, **layout):
        """Plot colors with labels

        Parameters
        ----------
        colors : dict
            Colors for cells.
        cells : tuple
            Cells for which to plot colors (default is ``colors.keys()``).
        """
        _EelFigure.__init__(self, "Colors", None, 2, 1.5, layout)

        if cells is None:
            cells = colors.keys()

        ax = self.figure.add_axes((0, 0, 1, 1), frameon=False)
        ax.set_axis_off()

        n = len(cells)
        for i, cell in enumerate(cells):
            bottom = n - i - 1
            y = bottom + 0.5
            patch = mpl.patches.Rectangle((0, bottom), 1, 1, fc=colors[cell],
                                          ec='none', zorder=1)
            ax.add_patch(patch)
            ax.text(1.1, y, cellname(cell), va='center', ha='left', zorder=2)

        ax.set_ylim(0, n)
        ax.set_xlim(0, n * self._layout.w / self._layout.h)

        self._show()

