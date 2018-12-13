# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from eelbrain import datasets, plot, NDVar
from eelbrain._wxgui.testing import hide_plots
from eelbrain.plot._glassbrain import _fast_abs_percentile

@hide_plots
def test_glassbrain():
    ndvar = datasets.get_mne_stc(True, True)

    # source space only
    p = plot.GlassBrain(ndvar.source, show=False)
    p.close()

    # single time points
    p = plot.GlassBrain(ndvar.sub(time=0.030), show=False)
    p.close()

    # time series
    p = plot.GlassBrain(ndvar.sub(time=0.030), show=False)
    p.close()

    # masked data
    import numpy as np
    h = ndvar.sub(time=0.030)
    c = _fast_abs_percentile(h)
    mask = h.norm('space') < c
    mask_x = np.repeat(h._ialign(mask), 3, h.get_axis('space'))
    mask = NDVar(mask_x, h.dims)
    y = h.mask(mask)
    p = plot.GlassBrain(y, show=False)
    p.close()
