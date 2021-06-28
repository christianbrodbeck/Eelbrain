# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from eelbrain import datasets, plot, boosting, epoch_impulse_predictor
from eelbrain.testing import hide_plots


@hide_plots
def test_datasplit():
    ds = datasets.get_uts()
    ds['x'] = epoch_impulse_predictor('uts', ds=ds)

    trf = boosting('uts', 'x', 0, 0.100, ds=ds, partitions=4)
    p = trf.splits.plot()
    p.close()

    trf = boosting('uts', 'x', 0, 0.100, ds=ds, partitions=4, test=True)
    p = trf.splits.plot()
    p.close()
