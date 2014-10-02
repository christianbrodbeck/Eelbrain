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

