# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from typing import Sequence, Tuple

from .._data_obj import Dataset, VarArg
from .. import fmtxt, plot


def scatter_table(
        xs: Sequence[VarArg],
        color: VarArg = None,
        ds: Dataset = None,
        diagonal: Tuple[float, float] = None,
        markers: str = '.',
        alpha: float = 0.5,
        **kwargs,
) -> fmtxt.Table:
    """Table with pairwise scatter-plots (see :class:`plot.Scatter`)

    Parameters
    ----------
    xs
        List of variables for pairwise comparisons.
    color
        Plot the correlation separately for different categories.
    ds
        If a Dataset is specified, all data-objects can be specified as
        names of Dataset variables
    diagonal
        Lower and upper value for plotting a diagonal (e.g. ``(0, 10)`` to plot
        diagonal from ``(0, 0)`` to ``(10, 10)``).
    **
        Other :class:`plot.Scatter` arguments.
    """
    table = fmtxt.Table('l' * (len(xs)-1))
    for ir, y in enumerate(xs):
        for ic, x in enumerate(xs):
            if ic == 0:
                continue
            elif ic == 1 and ir == len(xs) - 2 and color is not None:
                table.cell(p.plot_colorbar(orientation='vertical', h=p._layout.h, width=0.2).image())
                continue
            elif ic <= ir:
                table.cell()
                continue
            p = plot.Scatter(y, x, color, markers=markers, ds=ds, alpha=alpha, **kwargs)
            if diagonal:
                p._axes[0].plot(diagonal, diagonal, color='k')
            table.cell(p.image(format='png'))
        table.endline()
    return table
