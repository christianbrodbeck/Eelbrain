"""Test mne interaction"""
from itertools import izip
import os

from nose.tools import eq_, ok_, assert_less_equal, assert_not_equal, assert_in
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

import mne

from eelbrain import datasets, load, testnd, morph_source_space, Factor
from eelbrain._data_obj import asndvar, SourceSpace, _matrix_graph
from eelbrain._mne import shift_mne_epoch_trigger, combination_label
from eelbrain.tests.test_data import assert_dataobj_equal

# mne paths
data_dir = mne.datasets.sample.data_path()
subjects_dir = os.path.join(data_dir, 'subjects')


def test_source_estimate():
    "Test SourceSpace dimension"
    mne.set_log_level('warning')
    ds = datasets.get_mne_sample(src='ico')
    dsa = ds.aggregate('side')

    # test auto-conversion
    asndvar('epochs', ds=ds)
    asndvar('epochs', ds=dsa)
    asndvar(dsa['epochs'][0])

    # source space clustering
    res = testnd.ttest_ind('src', 'side', ds=ds, samples=0, pmin=0.05,
                           tstart=0.05, mintime=0.02, minsource=10)
    eq_(res.clusters.n_cases, 52)

    # test disconnecting parc
    src = ds['src']
    source = src.source
    parc = source.parc
    orig_conn = set(map(tuple, source.connectivity()))
    disc_conn = set(map(tuple, source.connectivity(True)))
    ok_(len(disc_conn) < len(orig_conn))
    for pair in orig_conn:
        s, d = pair
        if pair in disc_conn:
            eq_(parc[s], parc[d])
        else:
            assert_not_equal(parc[s], parc[d])

    # threshold-based test with parc
    srcl = src.sub(source='lh')
    res = testnd.ttest_ind(srcl, 'side', ds=ds, samples=10, pmin=0.05,
                           tstart=0.05, mintime=0.02, minsource=10,
                           parc='source')
    eq_(res._cdist.dist.shape[1], len(srcl.source.parc.cells))
    label = 'superiortemporal-lh'
    c_all = res.find_clusters(maps=True)
    c_label = res.find_clusters(maps=True, source=label)
    assert_array_equal(c_label['location'], label)
    for case in c_label.itercases():
        id_ = case['id']
        idx = c_all['id'].index(id_)[0]
        eq_(case['v'], c_all[idx, 'v'])
        eq_(case['tstart'], c_all[idx, 'tstart'])
        eq_(case['tstop'], c_all[idx, 'tstop'])
        assert_less_equal(case['p'], c_all[idx, 'p'])
        assert_dataobj_equal(case['cluster'],
                             c_all[idx, 'cluster'].sub(source=label))

    # threshold-free test with parc
    res = testnd.ttest_ind(srcl, 'side', ds=ds, samples=10, tstart=0.05,
                           parc='source')
    cl = res.find_clusters(0.05)
    eq_(cl.eval("p.min()"), res.p.min())
    mp = res.masked_parameter_map()
    assert_in(mp.min(), (0, res.t.min()))
    assert_in(mp.max(), (0, res.t.max()))

    # indexing source space
    s_sub = src.sub(source='fusiform-lh')
    idx = source.index_for_label('fusiform-lh')
    s_idx = src[idx]
    assert_dataobj_equal(s_sub, s_idx)


def test_dataobjects():
    "Test handing MNE-objects as data-objects"
    ds = datasets.get_mne_sample(sns=True)
    ds['C'] = Factor(ds['index'] > 155, labels={False: 'a', True: 'b'})
    sds = ds.sub("side % C != ('L', 'b')")
    ads = sds.aggregate('side % C')
    eq_(ads.n_cases, 3)

    # connectivity
    sensor = ds['sns'].sensor
    c = sensor.connectivity()
    assert_array_equal(c[:, 0] < c[:, 1], True)
    eq_(c.max(), len(sensor) - 1)


def test_epoch_trigger_shift():
    "Test the shift_mne_epoch_trigger() function"
    epochs = datasets.get_mne_sample(sns=True, sub="[1,2,3]")['epochs']
    n_lost_start = np.sum(epochs.times < epochs.tmin + 0.05)
    n_lost_end = np.sum(epochs.times > epochs.tmax - 0.05)
    data = epochs.get_data()

    epochs_s = shift_mne_epoch_trigger(epochs, [0, 0, 0])
    assert_array_equal(epochs_s.get_data(), data)

    epochs_s = shift_mne_epoch_trigger(epochs, [-0.05, 0., 0.05])
    data_s = epochs_s.get_data()
    assert_array_equal(data_s[0], data[0, :, : -(n_lost_end + n_lost_start)])
    assert_array_equal(data_s[1], data[1, :, n_lost_start: -n_lost_end])
    assert_array_equal(data_s[2], data[2, :, n_lost_end + n_lost_start:])
    assert_allclose(epochs_s.times, epochs.times[n_lost_start: -n_lost_end],
                    rtol=1e-1, atol=1e-3)  # ms accuracy

    epochs_s = shift_mne_epoch_trigger(epochs, [0.05, 0., 0.05])
    data_s = epochs_s.get_data()
    assert_array_equal(data_s[0], data[0, :, n_lost_end:])
    assert_array_equal(data_s[1], data[1, :, :-n_lost_end])
    assert_array_equal(data_s[2], data[2, :, n_lost_end:])
    assert_allclose(epochs_s.times, epochs.times[:-n_lost_end],
                    rtol=1e-1, atol=1e-3)  # ms accuracy


def test_combination_label():
    "Test combination label creation"
    labels = {l.name: l for l in
              mne.read_labels_from_annot('fsaverage', subjects_dir=subjects_dir)}

    # standard
    l = combination_label('temporal', "superiortemporal + middletemporal + inferiortemporal", labels)
    lh = labels['superiortemporal-lh'] + labels['middletemporal-lh'] + labels['inferiortemporal-lh']
    rh = labels['superiortemporal-rh'] + labels['middletemporal-rh'] + labels['inferiortemporal-rh']
    eq_(len(l), 2)
    eq_(l[0].name, 'temporal-lh')
    eq_(l[1].name, 'temporal-rh')
    assert_array_equal(l[0].vertices, lh.vertices)
    assert_array_equal(l[1].vertices, rh.vertices)

    # only rh
    l = combination_label('temporal-rh', "superiortemporal + middletemporal + inferiortemporal", labels)
    eq_(len(l), 1)
    eq_(l[0].name, 'temporal-rh')
    assert_array_equal(l[0].vertices, rh.vertices)

    # names with .
    labels = {l.name: l for l in
              mne.read_labels_from_annot('fsaverage', 'PALS_B12_Brodmann', subjects_dir=subjects_dir)}
    l = combination_label('Ba38-lh', "Brodmann.38", labels)[0]
    assert_array_equal(l.vertices, labels['Brodmann.38-lh'].vertices)


def test_morphing():
    mne.set_log_level('warning')
    sss = datasets._mne_source_space('fsaverage', 'ico-4', subjects_dir)
    vertices_to = [sss[0]['vertno'], sss[1]['vertno']]
    ds = datasets.get_mne_sample(-0.1, 0.1, src='ico', sub='index==0', stc=True)
    stc = ds['stc', 0]
    morph_mat = mne.compute_morph_matrix('sample', 'fsaverage', stc.vertno,
                                         vertices_to, None, subjects_dir)
    ndvar = ds['src']

    morphed_ndvar = morph_source_space(ndvar, 'fsaverage')
    morphed_stc = mne.morph_data_precomputed('sample', 'fsaverage', stc,
                                             vertices_to, morph_mat)
    assert_array_equal(morphed_ndvar.x[0], morphed_stc.data)
    morphed_stc_ndvar = load.fiff.stc_ndvar([morphed_stc], 'fsaverage', 'ico-4',
                                            subjects_dir, 'dSPM', False, 'src',
                                            parc=None)
    assert_dataobj_equal(morphed_ndvar, morphed_stc_ndvar)


def test_source_space():
    "Test SourceSpace dimension"
    for subject in ['fsaverage', 'sample']:
        mne_src = datasets._mne_source_space(subject, 'ico-4', subjects_dir)
        vertno = [mne_src[0]['vertno'], mne_src[1]['vertno']]
        ss = SourceSpace(vertno, subject, 'ico-4', subjects_dir, 'aparc')

        # connectivity
        conn = ss.connectivity()
        mne_conn = mne.spatial_src_connectivity(mne_src)
        assert_array_equal(conn, _matrix_graph(mne_conn))

        # sub-space connectivity
        sssub = ss[ss.dimindex('superiortemporal-rh')]
        ss2 = SourceSpace(vertno, subject, 'ico-4', subjects_dir, 'aparc')
        ss2sub = ss2[ss2.dimindex('superiortemporal-rh')]
        assert_array_equal(sssub.connectivity(), ss2sub.connectivity())
