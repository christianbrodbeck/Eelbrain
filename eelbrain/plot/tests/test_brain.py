# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from eelbrain import datasets, plot


def test_plot_brain():
    """Test plot.brain plots"""
    src = datasets.get_mne_sample(src='ico', sub=[0])['src']

    p = plot.brain.dspm(src)
    cb = p.plot_colorbar(show=False)
    cb.close()

    p = plot.brain.dspm(src, hemi='lh')
    cb = p.plot_colorbar(show=False)
    cb.close()

    p = plot.brain.cluster(src, hemi='rh', views='parietal')
    cb = p.plot_colorbar(show=False)
    cb.close()
