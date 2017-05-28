# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from __future__ import print_function

from nose.tools import eq_

from eelbrain import datasets, plot
from eelbrain._utils.testing import requires_mne_sample_data


@requires_mne_sample_data
def test_plot_brain():
    """Test plot.brain plots"""
    src = datasets.get_mne_sample(src='ico', sub=[0])['src']

    # size
    b = plot.brain.brain(src.source, hemi='rh', w=400, h=300, mask=False)
    eq_(b.screenshot().shape, (300, 400, 3))
    b.set_size(200, 150)
    eq_(b.screenshot().shape, (150, 200, 3))
    b.close()
    # both hemispheres
    b = plot.brain.brain(src.source, w=600, h=300, mask=False)
    eq_(b.screenshot().shape, (300, 600, 3))
    b.set_size(400, 150)
    eq_(b.screenshot().shape, (150, 400, 3))
    b.close()

    # plot shortcuts
    p = plot.brain.dspm(src)
    cb = p.plot_colorbar(show=False)
    cb.close()
    p.close()

    p = plot.brain.dspm(src, hemi='lh')
    cb = p.plot_colorbar(show=False)
    cb.close()
    p.close()

    p = plot.brain.cluster(src, hemi='rh', views='parietal')
    cb = p.plot_colorbar(show=False)
    cb.close()
    p.close()

    image = plot.brain.bin_table(src, tstart=0.1, tstop=0.3, tstep=0.1)
    print(repr(image))
    print(image)

    # plot p-map
    pmap = src.abs()
    pmap /= src.max()
    p = plot.brain.p_map(pmap, src)
    cb = p.plot_colorbar(show=False)
    cb.close()
    p.close()
