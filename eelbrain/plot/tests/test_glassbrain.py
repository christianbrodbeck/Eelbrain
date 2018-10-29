# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from eelbrain import datasets, plot
from eelbrain._wxgui.testing import hide_plots


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
