"""
Modules that provide data-specific plotting functions.

:mod:`eelbrain.plot.brain`
    Plot brain surface data.

:mod:`~eelbrain.plot.sensors`
    plotting EEG/MEG sensor maps

:mod:`~eelbrain.plot.topo`:
    topographic plots (EEG/MEG)

:mod:`~eelbrain.plot.uts` (uniform time series):
    plots for uniform time series

:mod:`~eelbrain.plot.utsnd` (n-dimensional uniform time series):
    uniform time series data with multiple variables

:mod:`~eelbrain.plot.uv` (univariate):
    plots for univariate data (such as boxplots, ...)


Module General Parameters
-------------------------

Xax
^^^

Many plots have an ``Xax`` parameter. This parameter takes a categorial data
object. The data of the plotted variable will be split into the catories in
``Xax``, and for every cell in ``Xax`` a separate subplot will be plotted.

For example, while

    >>> plot.utsnd.butterfly('meg', ds=ds)

will create a single butterfly plot of the average response,

    >>> plot.utsnd.butterfly('meg', 'subject', ds=ds)

where ``'subject'`` is the ``Xax`` parameter, will create a separate subplot
for every subject with its average response.


Layout
^^^^^^

Most plots that also share certain layout keyword arguments. By default, all
those parameters are determined automatically, but individual values can be
specified manually by supplying them as keyword arguments.

h, w : scalar
    Height and width of the figure.
axh, axw : scalar
    Height and width of the axes.
nrow, ncol : None | int
    Limit number of rows/columns. If neither is specified, a square layout
    is produced
ax_aspect : scalar
    Width / height aspect of the axes.

Plots that do take those parameters can be identified by the ``**layout`` in
their function signature.

"""
from . import sensors
from . import topo
from . import uts
from . import utsnd
from . import uv

from ._base import configure_backend
