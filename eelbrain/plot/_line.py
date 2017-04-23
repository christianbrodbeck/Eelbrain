# -*- coding: utf-8 -*-
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"Line plots"
from __future__ import division

from itertools import cycle, izip, repeat

import numpy as np

from .._data_obj import ascategorial, asndvar, assub
from ._base import (
    EelFigure, Layout, LegendMixin, XAxisMixin, find_axis_params_data,
    frame_title)


class LineStack(LegendMixin, XAxisMixin, EelFigure):
    u"""Stack multiple lines vertically
    
    Parameters
    ----------
    y : NDVar
        Values to plot.
    x : cateorial
        Variable to aggregate cases into lines (default is to plot each line).
    sub : None | index array
        Only use a subset of the data provided.
    ds : None | Dataset
        If a Dataset is specified, all data-objects can be specified as
        names of Dataset variables.
    offset : float | str
        The distance between the baseline (y = 0) for the different lines. 
        Can be a string expressed as a function of y. For example, 
        ``'0.66 * max(y)'`` will offset each line by 0.66 times the maximum 
        value in ``y`` (after aggregating if ``x`` is specified). The default is
        ``'2/3 * max(y.max(), -y.min())'``.
    xlabel : bool | str
        X-axis label. By default the label is inferred from the data.
    ylabel : bool | str
        Y-axis label. By default the label is inferred from the data.
    colors : dict | sequence of colors
        Colors for the lines (default is all lines in black).
    ylabels : bool | dict | sequence of str
        Labels for the different lines.
    legend : str | int | 'fig' | None
        Matplotlib figure legend location argument or 'fig' to plot the
        legend in a separate figure.
        
    Notes
    -----
    Navigation:
     - ``←``: scroll left
     - ``→``: scroll right
     - ``home``: scroll to beginning
     - ``end``: scroll to end
     - ``f``: x-axis zoom in (reduce x axis range)
     - ``d``: x-axis zoom out (increase x axis range)
    """
    _name = "LineStack"

    def __init__(self, y, x=None, sub=None, ds=None,
                 offset='max(y.max(), -y.min())', xlabel=True, ylabel=True,
                 colors=None, ylabels=True, xdim=None, legend=None,
                 *args, **kwargs):
        sub = assub(sub, ds)
        y = asndvar(y, sub, ds)
        if x is not None:
            x = ascategorial(x, sub, ds)
            y = y.aggregate(x)

        # find plotting dims
        if xdim is None and y.has_dim('time'):
            ydim, xdim = y.get_dimnames((None, 'time'))
        else:
            ydim, xdim = y.get_dimnames((None, xdim))
        xdim_obj = y.get_dim(xdim)

        # get data with offset
        ydata = y.get_data((ydim, xdim)).copy()
        ny = len(ydata)
        if isinstance(offset, basestring):
            offset = eval(offset, {'y': y})
        offset_a = np.arange(ny) * offset
        ydata += offset_a[:, np.newaxis]

        # find cells
        if x is None:
            if ydim == 'case':
                cells = range(ny)
            else:
                cells = y.get_dim(ydim)._as_uv()
            cell_labels = map(str, cells)
        else:
            cells = cell_labels = x.cells

        if ylabels is True:
            ylabels = cell_labels

        if ylabel is True:
            _, ylabel = find_axis_params_data(y, ylabel)

        # colors
        if colors is None:
            color_iter = repeat('k', ny)
        elif isinstance(colors, dict):
            color_iter = (colors[cell] for cell in cells)
        elif len(colors) < ny:
            color_iter = cycle(colors)
        else:
            color_iter = colors

        layout = Layout(1, 0.5, 6, *args, **kwargs)
        EelFigure.__init__(self, frame_title(y, x), layout)
        ax = self._axes[0]

        xdata = xdim_obj._axis_data()
        handles = [ax.plot(xdata, y_, color=color)[0] for y_, color in
                   izip(ydata, color_iter)]

        ax.grid(True)
        ax.set_frame_on(False)
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.set_yticks(offset_a)
        ax.set_yticklabels(ylabels or ())
        ax.set_ylim(y[0].min(), offset * ny)
        self._configure_xaxis_dim(xdim_obj, xlabel, True)
        if ylabel:
            ax.set_ylabel(ylabel)
        XAxisMixin.__init__(self, ((y,),), xdim)
        LegendMixin.__init__(self, legend, dict(izip(cell_labels, handles)))
        self._show()
