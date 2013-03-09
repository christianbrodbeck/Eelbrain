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

"""

# import _base
# figs = _base.figs
import sensors
import topo
import uts
import utsnd
import uv

from _base import unpack
