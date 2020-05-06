# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Style specification"""
from collections.abc import Iterator
from dataclasses import dataclass
from functools import reduce
from itertools import product
from math import ceil
import operator
from typing import Any, Dict

import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_rgba

from .._colorspaces import LocatedListedColormap, oneway_colors, twoway_colors, symmetric_cmaps
from .._data_obj import Factor, Interaction, CellArg
from .._utils import LazyProperty


StylesDict = Dict[CellArg, 'Style']


@dataclass
class Style:
    """Control color/pattern by category"""
    color: Any = (0, 0, 0)
    marker: str = None
    hatch: str = ''
    linestyle: str = None

    @LazyProperty
    def line_args(self):
        return {'color': self.color, 'linestyle': self.linestyle, 'marker': self.marker, 'markerfacecolor': self.color}

    @LazyProperty
    def patch_args(self):
        return {'facecolor': self.color, 'hatch': self.hatch}

    @classmethod
    def _coerce(cls, arg):
        if isinstance(arg, cls):
            return arg
        elif arg is None:
            return cls()
        else:
            return cls(arg)


def find_cell_styles(x, colors, cells=None) -> StylesDict:
    """Process the colors arg from plotting functions

    Parameters
    ----------
    x : categorial
        Model for which colors are needed. ``None`` if only a single value is
        plotted.
    colors : str | list | dict
        Colors for the plots if multiple categories of data are plotted.
        **str**: A colormap name; cells are mapped onto the colormap in
        regular intervals.
        **list**: A list of colors in the same sequence as cells.
        **dict**: A dictionary mapping each cell to a color.
        Colors are specified as `matplotlib compatible color arguments
        <http://matplotlib.org/api/colors_api.html>`_.
    cells : tuple of str
        In case only a subset of cells is used.
    """
    if x is None:
        if isinstance(colors, dict):
            color = colors[None]
        elif colors is None:
            color = 'k'
        else:
            color = to_rgba(colors)
        return {None: Style._coerce(color)}
    elif cells is None:
        cells = x.cells

    if isinstance(colors, (list, tuple)):
        if len(colors) < len(cells):
            raise ValueError(f"colors={colors!r}: only {len(colors)} colors for {len(cells)} cells.")
        out = dict(zip(cells, colors))
    elif isinstance(colors, dict):
        missing = [cell for cell in cells if cell not in colors]
        if missing:
            raise KeyError(f"colors={colors!r} is missing cells {missing}")
        out = colors
    elif colors is None or isinstance(colors, str):
        out = colors_for_categorial(x, cmap=colors)
    elif colors is False:
        return
    else:
        raise TypeError(f"colors={colors!r}")
    return {cell: Style._coerce(c) for cell, c in out.items()}


def colors_for_categorial(x, hue_start=0.2, cmap=None):
    """Automatically select colors for a categorial model

    Parameters
    ----------
    x : categorial
        Model defining the cells for which to define colors.
    hue_start : 0 <= scalar < 1
        First hue value (only for two-way or higher level models).
    cmap : str (optional)
        Name of a matplotlib colormap to use instead of default hue-based
        colors (only used for one-way models).

    Returns
    -------
    colors : dict {cell -> color}
        Dictionary providing colors for the cells in x.
    """
    if isinstance(x, Factor):
        return colors_for_oneway(x.cells, hue_start, cmap=cmap)
    elif isinstance(x, Interaction):
        return colors_for_nway([f.cells for f in x.base], hue_start)
    else:
        raise TypeError(f"x={x!r}: needs to be Factor or Interaction")


def colors_for_oneway(cells, hue_start=0.2, light_range=0.5, cmap=None,
                      light_cycle=None, always_cycle_hue=False, locations=None):
    """Define colors for a single factor design

    Parameters
    ----------
    cells : sequence of str
        Cells for which to assign colors.
    hue_start : 0 <= scalar < 1 | sequence of scalar
        First hue value (default 0.2) or list of hue values.
    light_range : scalar | tuple of 2 scalar
        Scalar that specifies the amount of lightness variation (default 0.5).
        If positive, the first color is lightest; if negative, the first color
        is darkest. A tuple can be used to specify exact end-points (e.g.,
        ``(1.0, 0.4)``). ``0.2`` is equivalent to ``(0.4, 0.6)``.
        The ``light_cycle`` parameter can be used to cycle between light and
        dark more than once.
    cmap : str
        Use a matplotlib colormap instead of the default color generation
        algorithm. Name of a matplotlib colormap to use (e.g., 'jet'). If
        specified, ``hue_start`` and ``light_range`` are ignored.
    light_cycle : int
        Cycle from light to dark in ``light_cycle`` cells to make nearby colors
        more distinct (default cycles once).
    always_cycle_hue : bool
        Cycle hue even when cycling lightness. With ``False`` (default), hue
        is constant within a lightness cycle.
    locations : sequence of float
        Locations of the cells on the color-map (all in range [0, 1]; default is
        evenly spaced; example: ``numpy.linspace(0, 1, len(cells)) ** 0.5``).

    Returns
    -------
    dict : {str: tuple}
        Mapping from cells to colors.
    """
    if isinstance(cells, Iterator):
        cells = tuple(cells)
    n = len(cells)
    if cmap is None:
        colors = oneway_colors(n, hue_start, light_range, light_cycle, always_cycle_hue, locations)
    else:
        cm = mpl.cm.get_cmap(cmap)
        if locations is None:
            imax = n - 1
            locations = (i / imax for i in range(n))
        colors = (cm(x) for x in locations)
    return dict(zip(cells, colors))


def colors_for_twoway(x1_cells, x2_cells, hue_start=0.2, hue_shift=0., hues=None, lightness=None):
    """Define cell colors for a two-way design

    Parameters
    ----------
    x1_cells : tuple of str
        Cells of the major factor.
    x2_cells : tuple of str
        Cells of the minor factor.
    hue_start : 0 <= scalar < 1
        First hue value.
    hue_shift : 0 <= scalar < 1
        Use that part of the hue continuum between categories to shift hue
        within categories.
    hues : list of scalar
        List of hue values corresponding to the levels of the first factor
        (overrides regular hue distribution).
    lightness : scalar | list of scalar
        If specified as scalar, colors will occupy the range
        ``[lightness, 100-lightness]``. Can also be given as list with one
        value corresponding to each element in the second factor.

    Returns
    -------
    dict : {tuple: tuple}
        Mapping from cells to colors.
    """
    n1 = len(x1_cells)
    n2 = len(x2_cells)

    if n1 < 2 or n2 < 2:
        raise ValueError("Need at least 2 cells on each factor")

    clist = twoway_colors(n1, n2, hue_start, hue_shift, hues, lightness)
    return dict(zip(product(x1_cells, x2_cells), clist))


def colors_for_nway(cell_lists, hue_start=0.2):
    """Define cell colors for a two-way design

    Parameters
    ----------
    cell_lists : sequence of of tuple of str
        List of the cells for each factor. E.g. for ``A % B``:
        ``[('a1', 'a2'), ('b1', 'b2', 'b3')]``.
    hue_start : 0 <= scalar < 1
        First hue value.

    Returns
    -------
    dict : {tuple: tuple}
        Mapping from cells to colors.
    """
    if len(cell_lists) == 1:
        return colors_for_oneway(cell_lists[0])
    elif len(cell_lists) == 2:
        return colors_for_twoway(cell_lists[0], cell_lists[1], hue_start)
    elif len(cell_lists) > 2:
        ns = tuple(map(len, cell_lists))
        n_outer = reduce(operator.mul, ns[:-1])
        n_inner = ns[-1]

        # outer circle
        hues = np.linspace(hue_start, 1 + hue_start, ns[0], False)
        # subdivide for each  level
        distance = 1. / ns[0]
        for n_current in ns[1:-1]:
            new = []
            d = distance / 3
            for hue in hues:
                new.extend(np.linspace(hue - d, hue + d, n_current))
            hues = new
            distance = 2 * d / (n_current - 1)
        hues = np.asarray(hues)
        hues %= 1
        colors = twoway_colors(n_outer, n_inner, hues=hues)
        return dict(zip(product(*cell_lists), colors))
    else:
        return {}


def single_hue_colormap(hue):
    """Colormap based on single hue

    Parameters
    ----------
    hue : matplotlib color
        Base RGB color.

    Returns
    -------
    colormap : matplotlib Colormap
        Colormap from transparent to ``hue``.
    """
    name = str(hue)
    color = to_rgb(hue)
    start = color + (0.,)
    stop = color + (1.,)
    return LinearSegmentedColormap.from_list(name, (start, stop))


def soft_threshold_colormap(cmap, threshold, vmax, subthreshold=None, symmetric=None):
    """Soft-threshold a colormap to make small values transparent

    Parameters
    ----------
    cmap : str
        Base colormap.
    threshold : scalar
        Value at which to threshold the colormap (i.e., the value at which to
        start the colormap).
    vmax : scalar
        Intended largest value of the colormap (used to infer the location of
        the ``threshold``).
    subthreshold : matplotlib color
        Color of sub-threshold values (the default is the end or middle of
        the colormap, depending on whether it is symmetric).
    symmetric : bool
        Whether the ``cmap`` is symmetric (ranging from ``-vmax`` to ``vmax``)
        or not (ranging from ``0`` to ``vmax``). The default is ``True`` for
        known symmetric colormaps and ``False`` otherwise.

    Returns
    -------
    thresholded_cmap : matplotlib ListedColormap
        Soft-thresholded colormap.
    """
    assert vmax > threshold >= 0
    cmap = mpl.cm.get_cmap(cmap)
    if symmetric is None:
        symmetric = cmap.name in symmetric_cmaps

    colors = cmap(np.linspace(0., 1., cmap.N))
    if subthreshold is None:
        subthreshold_color = cmap(0.5) if symmetric else cmap(0)
    else:
        subthreshold_color = to_rgba(subthreshold)

    n = int(round(vmax / ((vmax - threshold) / cmap.N)))
    out_colors = np.empty((n, 4))
    if symmetric:
        i_threshold = int(ceil(cmap.N / 2))
        out_colors[:i_threshold] = colors[:i_threshold]
        out_colors[i_threshold:-i_threshold] = subthreshold_color
        out_colors[-i_threshold:] = colors[-i_threshold:]
    else:
        out_colors[:-cmap.N] = subthreshold_color
        out_colors[-cmap.N:] = colors
    out = LocatedListedColormap(out_colors, cmap.name)
    out.vmax = vmax
    out.vmin = -vmax if symmetric else 0
    out.symmetric = symmetric
    return out
