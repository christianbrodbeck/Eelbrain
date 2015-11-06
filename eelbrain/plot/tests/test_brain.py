# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from eelbrain import datasets, plot


def test_plot_brain():
    """Test plot.brain plots"""
    src = datasets.get_mne_sample(src='ico', sub=[0])['src']

    p = plot.brain.dspm(src)
    cb = p.plot_colorbar(show=False)
    cb.close()
    p.close()
    # not closing figures leads to weird interactions with the QT backend

    p = plot.brain.dspm(src, hemi='lh')
    cb = p.plot_colorbar(show=False)
    cb.close()
    p.close()

    p = plot.brain.cluster(src, hemi='rh', views='parietal')
    cb = p.plot_colorbar(show=False)
    cb.close()
    p.close()

    image = plot.brain.bin_table(src, tstart=0.1, tstop=0.3, tstep=0.1)
    print repr(image)
    print image
