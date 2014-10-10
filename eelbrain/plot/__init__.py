"""Plotting for data-objects

.. _plotting-general:

Xax parameter
^^^^^^^^^^^^^

Many plots have an ``Xax`` parameter. This parameter takes a categorial data
object. The data of the plotted variable will be split into the catories in
``Xax``, and for every cell in ``Xax`` a separate subplot will be plotted.

For example, while

    >>> plot.Butterfly('meg', ds=ds)

will create a single Butterfly plot of the average response,

    >>> plot.Butterfly('meg', 'subject', ds=ds)

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


Backend Configuration
^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: eelbrain.plot

By default, Matplotlib-based Eelbrain figures (all except :mod:`.plot.brain`)
are opened in a wxPython based GUI application (see :ref:`gui`). By default,
this GUI is activated whenever a figure is created in interactive mode. These
defaults can be changed with :func:`configure_backend`:

.. autofunction:: configure_backend

"""
from ._colors import (colors_for_categorial, colors_for_oneway,
                      colors_for_twoway, ColorBar, ColorGrid, ColorList)
from ._sensors import SensorMaps, SensorMap2d
from ._topo import TopoArray, TopoButterfly, Topomap
from ._uts import UTSStat, UTS, UTSClusters
from ._utsnd import Array, Butterfly
from ._uv import (Barplot, Boxplot, Correlation, Histogram, PairwiseLegend,
                  Regression, Timeplot)
from . import brain

from ._base import configure_backend
