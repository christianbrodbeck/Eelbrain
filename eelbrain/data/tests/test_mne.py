"""Test mne interaction"""
import os

from nose.tools import assert_equal, assert_not_equal, assert_true
from numpy.testing import assert_array_equal

import mne

from eelbrain.lab import datasets, testnd, morph_source_space
from eelbrain.data.data_obj import asndvar
from eelbrain.data.tests.test_data import assert_dataobj_equal


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
    assert_equal(res.clusters.n_cases, 52)

    # test disconnecting parc
    src = ds['src']
    source = src.source
    parc = source.parc
    orig_conn = set(map(tuple, source.connectivity()))
    disc_conn = set(map(tuple, source.connectivity(True)))
    assert_true(len(disc_conn) < len(orig_conn))
    for pair in orig_conn:
        s, d = pair
        if pair in disc_conn:
            assert_equal(parc[s], parc[d])
        else:
            assert_not_equal(parc[s], parc[d])

    # indexing source space
    s_sub = src.sub(source='fusiform-lh')
    idx = source.index_for_label('fusiform-lh')
    s_idx = src[idx]
    assert_dataobj_equal(s_sub, s_idx)

    # test morphing
    dsa = ds.aggregate('side')
    ndvar = dsa['src']
    stc = mne.SourceEstimate(ndvar.x[0], ndvar.source.vertno,
                             ndvar.time.tmin, ndvar.time.tstep,
                             ndvar.source.subject)
    subjects_dir = ndvar.source.subjects_dir
    path = ndvar.source._src_pattern.format(subject='fsaverage',
                                            src=ndvar.source.src,
                                            subjects_dir=subjects_dir)
    if os.path.exists(path):
        src_to = mne.read_source_spaces(path)
    else:
        src_to = mne.setup_source_space('fsaverage', path, 'ico4',
                                        subjects_dir=subjects_dir)
    vertices_to = [src_to[0]['vertno'], src_to[1]['vertno']]
    mm = mne.compute_morph_matrix('sample', 'fsaverage', ndvar.source.vertno,
                                  vertices_to, None, subjects_dir)
    stc_to = mne.morph_data_precomputed('sample', 'fsaverage', stc,
                                        vertices_to, mm)

    ndvar_m = morph_source_space(ndvar, 'fsaverage')
    assert_array_equal(ndvar_m.x[0], stc_to.data)
