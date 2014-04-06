"""Test mne interaction"""
import os

from nose.tools import assert_equal
from numpy.testing import assert_array_equal

import mne

from eelbrain.lab import datasets, testnd, morph_source_space


def test_source_estimate():
    "Test SourceSpace dimension"
    ds = datasets.get_mne_sample(src='ico')

    # source space clustering
    res = testnd.ttest_ind('src', 'side', ds=ds, samples=0, pmin=0.05,
                           tstart=0.05, mintime=0.02, minsource=10)
    assert_equal(res.clusters.n_cases, 52)

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
