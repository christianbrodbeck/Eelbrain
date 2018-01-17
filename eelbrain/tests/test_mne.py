"""Test mne interaction"""
from itertools import izip
import os

from nose.tools import (
    eq_, ok_, assert_almost_equal, assert_less_equal, assert_not_equal,
    assert_in, assert_is, assert_less, assert_is_instance,
)
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

import mne
from mne.tests.test_label import assert_labels_equal
from nibabel.freesurfer import read_annot

from eelbrain import (
    datasets, load, testnd,
    Dataset, Factor,
    concatenate, labels_from_clusters, morph_source_space, set_parc, xhemi)
from eelbrain._data_obj import SourceSpace, asndvar, _matrix_graph
from eelbrain._mne import shift_mne_epoch_trigger, combination_label
from eelbrain._utils.testing import requires_mne_sample_data
from eelbrain.tests.test_data import assert_dataobj_equal

data_dir = mne.datasets.testing.data_path()
subjects_dir = os.path.join(data_dir, 'subjects')


def assert_label_equal(l1, l2):
    if isinstance(l1, mne.BiHemiLabel):
        assert_is_instance(l2, mne.BiHemiLabel)
        assert_labels_equal(l1.lh, l2.lh)
        assert_labels_equal(l1.rh, l2.rh)
    else:
        assert_labels_equal(l1, l2)


@requires_mne_sample_data
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

    # concatenate
    src_reconc = concatenate((src.sub(source='lh'), src.sub(source='rh')), 'source')
    assert_dataobj_equal(src_reconc, src)


@requires_mne_sample_data
def test_dataobjects():
    "Test handing MNE-objects as data-objects"
    shift = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.1, -0.1])
    epochs = datasets.get_mne_epochs()
    ds = Dataset(('a', Factor('ab', repeat=8)),
                 ('epochs', epochs))
    ds['ets'] = shift_mne_epoch_trigger(epochs, shift, min(shift), max(shift))

    # ds operations
    sds = ds.sub("a == 'a'")
    ads = ds.aggregate('a')

    # asndvar
    ndvar = asndvar(ds['epochs'])
    ndvar = asndvar(ds['ets'])

    # connectivity
    ds = datasets.get_mne_sample(sub=[0], sns=True)
    sensor = ds['meg'].sensor
    c = sensor.connectivity()
    assert_array_equal(c[:, 0] < c[:, 1], True)
    eq_(c.max(), len(sensor) - 1)


@requires_mne_sample_data
def test_epoch_trigger_shift():
    "Test the shift_mne_epoch_trigger() function"
    epochs = datasets.get_mne_sample(sns=True, sub="[1,2,3]")['epochs']
    epochs.info['projs'] = []
    n_lost_start = np.sum(epochs.times < epochs.tmin + 0.05)
    n_lost_end = np.sum(epochs.times > epochs.tmax - 0.05)
    data = epochs.get_data()

    # don't shift
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
    l = combination_label('temporal', "superiortemporal + middletemporal + inferiortemporal",
                          labels, subjects_dir)
    lh = labels['superiortemporal-lh'] + labels['middletemporal-lh'] + labels['inferiortemporal-lh']
    lh.name = 'temporal-lh'
    rh = labels['superiortemporal-rh'] + labels['middletemporal-rh'] + labels['inferiortemporal-rh']
    rh.name = 'temporal-rh'
    eq_(len(l), 2)
    assert_labels_equal(l[0], lh)
    assert_labels_equal(l[1], rh)

    # only rh
    l = combination_label('temporal-rh', "superiortemporal + middletemporal + inferiortemporal",
                          labels, subjects_dir)
    eq_(len(l), 1)
    eq_(l[0].name, 'temporal-rh')
    assert_array_equal(l[0].vertices, rh.vertices)

    # with split_label
    l2 = combination_label('temporal-rh', "superiortemporal + middletemporal +"
                                          "split(inferiortemporal, 2)[0] +"
                                          "split(inferiortemporal, 2)[1]",
                           labels, subjects_dir)
    assert_labels_equal(l2[0], l[0], comment=False, color=False)

    # names with .
    labels = {l.name: l for l in
              mne.read_labels_from_annot('fsaverage', 'PALS_B12_Lobes', subjects_dir=subjects_dir)}
    l = combination_label('frontal-lh', "LOBE.FRONTAL", labels, subjects_dir)[0]
    assert_array_equal(l.vertices, labels['LOBE.FRONTAL-lh'].vertices)


@requires_mne_sample_data
def test_morphing():
    mne.set_log_level('warning')
    data_dir = mne.datasets.sample.data_path()
    subjects_dir = os.path.join(data_dir, 'subjects')
    sss = datasets._mne_source_space('fsaverage', 'ico-4', subjects_dir)
    vertices_to = [sss[0]['vertno'], sss[1]['vertno']]
    ds = datasets.get_mne_sample(-0.1, 0.1, src='ico', sub='index==0', stc=True)

    stc = ds['stc', 0]
    morph_mat = mne.compute_morph_matrix('sample', 'fsaverage', stc.vertices,
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

    # xhemi
    lh, rh = xhemi(ndvar, mask=False)
    eq_(lh.source.rh_n, 0)
    eq_(rh.source.rh_n, 0)
    assert_almost_equal(lh.max(), 19.63, 2)
    assert_almost_equal(rh.max(), 22.79, 2)


@requires_mne_sample_data  # source space distance computation times out
def test_source_space():
    "Test SourceSpace dimension"
    data_dir = mne.datasets.sample.data_path()
    subjects_dir = os.path.join(data_dir, 'subjects')
    annot_path = os.path.join(subjects_dir, '%s', 'label', '%s.%s.annot')

    for subject in ['fsaverage', 'sample']:
        mne_src = datasets._mne_source_space(subject, 'ico-4', subjects_dir)
        vertices = [mne_src[0]['vertno'], mne_src[1]['vertno']]
        ss = SourceSpace(vertices, subject, 'ico-4', subjects_dir)

        # labels
        for hemi_vertices, hemi in izip(ss.vertices, ('lh', 'rh')):
            labels, _, names = read_annot(annot_path % (subject, hemi, 'aparc'))
            start = 0 if hemi == 'lh' else len(ss.lh_vertices)
            hemi_tag = '-' + hemi
            for i, v in enumerate(hemi_vertices, start):
                label = labels[v]
                if label == -1:
                    eq_(ss.parc[i], 'unknown' + hemi_tag)
                else:
                    eq_(ss.parc[i], names[label] + hemi_tag)

        # connectivity
        conn = ss.connectivity()
        mne_conn = mne.spatial_src_connectivity(mne_src)
        assert_array_equal(conn, _matrix_graph(mne_conn))

        # sub-space connectivity
        sssub = ss[ss._array_index('superiortemporal-rh')]
        ss2 = SourceSpace(vertices, subject, 'ico-4', subjects_dir, 'aparc')
        ss2sub = ss2[ss2._array_index('superiortemporal-rh')]
        assert_array_equal(sssub.connectivity(), ss2sub.connectivity())


@requires_mne_sample_data
def test_source_ndvar():
    "Test NDVar with source dimension"
    ds = datasets.get_mne_sample(-0.1, 0.1, src='ico', sub='index<=1')
    v = ds['src', 0]
    eq_(v.source.parc.name, 'aparc')
    v_2009 = set_parc(v, 'aparc.a2009s')
    eq_(v_2009.source.parc.name, 'aparc.a2009s')
    conn = v_2009.source.connectivity()
    assert_less(np.sum(v.source.parc == v_2009.source.parc), len(v.source))
    v_back = set_parc(v_2009, 'aparc')
    eq_(v_back.source.parc.name, 'aparc')
    assert_array_equal(v.source.parc, v_back.source.parc)
    assert_is(v.x, v_back.x)
    assert_array_equal(v_back.source.connectivity(), conn)

    # labels_from_cluster
    v1, v2 = ds['src']
    v1 = v1 * (v1 > 15)
    labels1 = labels_from_clusters(v1)
    eq_(len(labels1), 1)
    labels1s = labels_from_clusters(v1.sum('time'))
    eq_(len(labels1s), 1)
    assert_label_equal(labels1s[0], labels1[0])
    v2 = v2 * (v2 > 2)
    labels2 = labels_from_clusters(concatenate((v1, v2), 'case'))
    eq_(len(labels2), 2)
    assert_label_equal(labels1[0], labels2[0])
