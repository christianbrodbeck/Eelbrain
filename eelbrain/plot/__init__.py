"""Plotting for data-objects"""
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>

from .._colorspaces import two_step_colormap

from ._base import reset_rc
from ._colors import ColorBar, ColorGrid, ColorList
from ._figure import Figure
from ._glassbrain import GlassBrain
from ._line import LineStack
from ._sensors import SensorMaps, SensorMap
from ._styles import Style, colors_for_categorial, colors_for_oneway, colors_for_twoway, single_hue_colormap, soft_threshold_colormap
from ._topo import TopoArray, TopoButterfly, Topomap, TopomapBins
from ._split import DataSplit
from ._uts import UTSStat, UTS, UTSClusters
from ._utsnd import Array, Butterfly
from ._uv import Barplot, BarplotHorizontal, Boxplot, Scatter, Histogram, PairwiseLegend, Regression, Timeplot
from . import brain
