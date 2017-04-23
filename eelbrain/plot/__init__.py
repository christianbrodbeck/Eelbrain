"""Plotting for data-objects"""
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>

from ._colors import (colors_for_categorial, colors_for_oneway,
                      colors_for_twoway, ColorBar, ColorGrid, ColorList)
from ._line import LineStack
from ._sensors import SensorMaps, SensorMap
from ._topo import TopoArray, TopoButterfly, Topomap, TopomapBins
from ._uts import UTSStat, UTS, UTSClusters
from ._utsnd import Array, Butterfly
from ._uv import (Barplot, Boxplot, Correlation, Histogram, PairwiseLegend,
                  Regression, Timeplot)
from . import brain


def configure(*args, **kwargs):
    raise RuntimeError("This function has been removed. Please use "
                       "eelbrain.configure() instead")
