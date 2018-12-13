# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from eelbrain import datasets, plot
from eelbrain._wxgui.testing import hide_plots


@hide_plots
def test_glassbrain():
    ndvar = datasets.get_mne_stc(True, True)

    # source space only
    p = plot.GlassBrain(ndvar.source)
    p.close()

    # single time points
    ndvar_30 = ndvar.sub(time=0.030)
    p = plot.GlassBrain(ndvar_30)
    p.close()
    # without arrows
    p = plot.GlassBrain(ndvar_30, draw_arrows=False)
    p.close()

    # time series
    p = plot.GlassBrain(ndvar)
    p.set_time(.03)
    p.close()
