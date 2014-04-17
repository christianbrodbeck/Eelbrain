# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>

import cPickle as pickle
import logging

from nose.tools import (assert_equal, assert_in, assert_less, assert_not_in,
                        assert_raises)
import numpy as np
from numpy.testing import assert_array_equal

from eelbrain.data.data_obj import UTS, Ordered, Sensor
from eelbrain.data.stats.testnd import _ClusterDist
from eelbrain.data.tests.test_data import assert_dataobj_equal
from eelbrain.data import datasets, testnd, plot, NDVar


def test_anova():
    "Test testnd.anova()"
    plot.configure_backend(False, False)
    ds = datasets.get_rand(True)

    testnd.anova('utsnd', 'A*B', ds=ds)

    res = testnd.anova('utsnd', 'A*B*rm', ds=ds)
    repr(res)
    p = plot.Array(res)
    p.close()

    res = testnd.anova('utsnd', 'A*B*rm', ds=ds, samples=2, pmin=0.05)
    repr(res)
    p = plot.Array(res)
    p.close()

    # persistence
    string = pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL)
    res_ = pickle.loads(string)
    assert_equal(repr(res_), repr(res))

    # test multi-effect results (with persistence)
    # UTS
    res = testnd.anova('uts', 'A*B*rm', ds=ds, samples=5)
    repr(res)
    string = pickle.dumps(res, pickle.HIGHEST_PROTOCOL)
    res = pickle.loads(string)
    tfce_clusters = res.tfce_clusters(pmin=0.05)
    peaks = res.tfce_peaks()
    assert_equal(tfce_clusters.eval("p.min()"), peaks.eval("p.min()"))
    unmasked = res.f[0]
    masked = res.masked_parameter_map(effect=0, pmin=0.05)
    assert_array_equal(masked.x <= unmasked.x, True)


def test_clusterdist():
    "Test _ClusterDist class"
    shape = (10, 6, 6, 4)
    locs = [[0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0]]
    x = np.random.normal(0, 1, shape)
    dims = ('case', UTS(-0.1, 0.1, 6), Ordered('dim2', range(6), 'unit'),
            Sensor(locs, ['0', '1', '2', '3'], connect_dist=1.1))
    Y = NDVar(x, dims)

    # test connecting sensors
    bin_map = np.zeros(shape[1:], dtype=np.bool8)
    bin_map[:3, :3, :2] = True
    pmap = np.random.normal(0, 1, shape[1:])
    np.clip(pmap, -1, 1, pmap)
    pmap[bin_map] = 2
    cdist = _ClusterDist(Y, 0, 1.5)
    cdist.add_original(pmap)
    assert_equal(cdist.n_clusters, 1)
    assert_array_equal(cdist._original_cluster_map == cdist._cids[0],
                       cdist._crop(bin_map).swapaxes(0, cdist._nad_ax))
    assert_equal(cdist.parameter_map.dims, Y.dims[1:])

    # test connecting many sensors
    bin_map = np.zeros(shape[1:], dtype=np.bool8)
    bin_map[:3, :3] = True
    pmap = np.random.normal(0, 1, shape[1:])
    np.clip(pmap, -1, 1, pmap)
    pmap[bin_map] = 2
    cdist = _ClusterDist(Y, 0, 1.5)
    cdist.add_original(pmap)
    assert_equal(cdist.n_clusters, 1)
    assert_array_equal(cdist._original_cluster_map == cdist._cids[0],
                       cdist._crop(bin_map).swapaxes(0, cdist._nad_ax))

    # test keeping sensors separate
    bin_map = np.zeros(shape[1:], dtype=np.bool8)
    bin_map[:3, :3, 0] = True
    bin_map[:3, :3, 2] = True
    pmap = np.random.normal(0, 1, shape[1:])
    np.clip(pmap, -1, 1, pmap)
    pmap[bin_map] = 2
    cdist = _ClusterDist(Y, 1, 1.5)
    cdist.add_original(pmap)
    assert_equal(cdist.n_clusters, 2)

    # TFCE
    dims = ('case', UTS(-0.1, 0.1, 4),
            Sensor(locs, ['0', '1', '2', '3'], connect_dist=1.1),
            Ordered('dim2', range(10), 'unit'))
    Y = NDVar(np.random.normal(0, 1, (10, 4, 4, 10)), dims)
    cdist = _ClusterDist(Y, 3, None)
    cdist.add_original(Y.x[0])
    for i in xrange(1, 4):
        cdist.add_perm(Y.x[i])
    # I/O
    string = pickle.dumps(cdist, pickle.HIGHEST_PROTOCOL)
    cdist_ = pickle.loads(string)
    assert_equal(repr(cdist_), repr(cdist))
    # find peaks
    x = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [7, 7, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 7, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [5, 7, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 6, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 7, 5, 5, 0, 0],
                   [0, 0, 0, 0, 5, 4, 4, 4, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
                   [0, 0, 0, 0, 7, 0, 0, 3, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]])
    tgt = np.equal(x, 7)
    peaks = cdist._find_peaks(x)
    logging.debug(' detected: \n%s' % (peaks.astype(int)))
    logging.debug(' target: \n%s' % (tgt.astype(int)))
    assert_array_equal(peaks, tgt)

    # test keeping dimension
    assert_equal(cdist.dist.shape, (3,))
    cdist = _ClusterDist(Y, 5, None, dist_dim='sensor')
    cdist.add_original(Y.x[0])
    for i in xrange(1, 6):
        cdist.add_perm(Y.x[i])
    assert_equal(cdist.dist.shape, (5, 4))

    # test keeping time bins
    cdist = _ClusterDist(Y, 5, None, dist_tstep=0.2)
    cdist.add_original(Y.x[0])
    for i in xrange(1, 6):
        cdist.add_perm(Y.x[i])
    assert_equal(cdist.dist.shape, (5, 2))
    assert_raises(ValueError, _ClusterDist, Y, 5, None, dist_tstep=0.3)

    # test keeping dimension and time bins
    cdist = _ClusterDist(Y, 5, None, dist_dim='sensor', dist_tstep=0.2)
    cdist.add_original(Y.x[0])
    for i in xrange(1, 6):
        cdist.add_perm(Y.x[i])
    assert_equal(cdist.dist.shape, (5, 4, 2))

    # test keeping 2 dimensions and time bins
    cdist = _ClusterDist(Y, 5, None, dist_dim=('sensor', 'dim2'),
                         dist_tstep=0.2)
    cdist.add_original(Y.x[0])
    for i in xrange(1, 6):
        cdist.add_perm(Y.x[i])
    assert_equal(cdist.dist.shape, (5, 4, 2, 10))


def test_corr():
    "Test testnd.corr()"
    plot.configure_backend(False, False)
    ds = datasets.get_rand(True)

    # add correlation
    Y = ds['Y']
    utsnd = ds['utsnd']
    utsnd.x.shape
    utsnd.x[:, 3:5, 50:65] += Y.x[:, None, None]

    res = testnd.corr('utsnd', 'Y', 'rm', ds=ds)
    repr(res)
    p = plot.Array(res)
    p.close()

    res = testnd.corr('utsnd', 'Y', 'rm', ds=ds, samples=10, pmin=0.05)
    p = plot.Array(res)
    p.close()

    # persistence
    string = pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL)
    res_ = pickle.loads(string)
    assert_equal(repr(res_), repr(res))
    assert_dataobj_equal(res.p_uncorrected, res_.p_uncorrected)
    assert_dataobj_equal(res.p, res_.p)


def test_t_contrast():
    ds = datasets.get_rand()

    # simple contrast
    res = testnd.t_contrast_rel('uts', 'A', 'a1>a0', 'rm', ds=ds, samples=100,
                                pmin=0.05)
    repr(res)
    res_ = testnd.ttest_rel('uts', 'A', 'a1', 'a0', 'rm', ds=ds)
    assert_array_equal(res.t.x, res_.t.x)
    assert_in('samples', repr(res))

    # complex contrast
    res = testnd.t_contrast_rel('uts', 'A%B', 'min(a0|b0>a1|b0, a0|b1>a1|b1)',
                                'rm', ds=ds, samples=100, pmin=0.05)
    res_b0 = testnd.ttest_rel('uts', 'A%B', ('a0', 'b0'), ('a1', 'b0'), 'rm',
                              ds=ds)
    res_b1 = testnd.ttest_rel('uts', 'A%B', ('a0', 'b1'), ('a1', 'b1'), 'rm',
                              ds=ds)
    assert_array_equal(res.t.x, np.min([res_b0.t.x, res_b1.t.x], axis=0))

    # persistence
    string = pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL)
    res_ = pickle.loads(string)
    assert_equal(repr(res_), repr(res))
    assert_dataobj_equal(res.p, res_.p)


def test_ttest_1samp():
    "Test testnd.ttest_1samp()"
    ds = datasets.get_rand(True)

    # no clusters
    res0 = testnd.ttest_1samp('uts', sub="A == 'a0'", ds=ds)
    assert_less(res0.p_uncorrected.min(), 0.05)
    repr0 = repr(res0)
    assert_in("'uts'", repr0)
    assert_not_in('clusters', repr0)
    assert_not_in('mintime', repr0)

    # clusters without resampling
    res1 = testnd.ttest_1samp('uts', sub="A == 'a0'", ds=ds, samples=0,
                              pmin=0.05, tstart=0, tstop=0.6, mintime=0.05)
    assert_equal(res1.clusters.n_cases, 2)
    assert_not_in('p', res1.clusters)
    repr1 = repr(res1)
    assert_in('clusters', repr1)
    assert_in('samples', repr1)
    assert_in('mintime', repr1)

    # persistence
    string = pickle.dumps(res1, pickle.HIGHEST_PROTOCOL)
    res1_ = pickle.loads(string)
    assert_equal(repr(res1_), repr1)
    assert_dataobj_equal(res1.p_uncorrected, res1_.p_uncorrected)

    # clusters with resampling
    res2 = testnd.ttest_1samp('uts', sub="A == 'a0'", ds=ds, samples=10,
                              pmin=0.05, tstart=0, tstop=0.6, mintime=0.05)
    assert_equal(res2.clusters.n_cases, 2)
    assert_equal(res2.samples, 10)
    assert_in('p', res2.clusters)
    repr2 = repr(res2)
    assert_in('samples', repr2)

    # clusters with permutations
    dss = ds.sub("logical_and(A=='a0', B=='b0')")[:8]
    res3 = testnd.ttest_1samp('uts', sub="A == 'a0'", ds=dss, samples=10000,
                              pmin=0.05, tstart=0, tstop=0.6, mintime=0.05)
    assert_equal(res3.clusters.n_cases, 2)
    assert_equal(res3.samples, -1)
    assert_less(res3.clusters['p'].x.min(), 0.05)
    repr3 = repr(res3)
    assert_in('samples', repr3)

    # TFCE properties
    res = testnd.ttest_1samp('utsnd', sub="A == 'a0'", ds=ds, samples=1)
    string = pickle.dumps(res, pickle.HIGHEST_PROTOCOL)
    res = pickle.loads(string)
    tfce_clusters = res.tfce_clusters(pmin=0.05)
    peaks = res.tfce_peaks()
    assert_equal(tfce_clusters.eval("p.min()"), peaks.eval("p.min()"))
    masked = res.masked_parameter_map(pmin=0.05)
    assert_array_equal(masked.abs().x <= res.t.abs().x, True)


def test_ttest_ind():
    "Test testnd.ttest_ind()"
    ds = datasets.get_rand()

    # basic
    res = testnd.ttest_ind('uts', 'A', 'a1', 'a0', ds=ds)
    repr(res)
    assert_less(res.p_uncorrected.min(), 0.05)
    # persistence
    string = pickle.dumps(res, pickle.HIGHEST_PROTOCOL)
    res_ = pickle.loads(string)
    repr(res_)
    assert_dataobj_equal(res.p_uncorrected, res_.p_uncorrected)

    # cluster
    res = testnd.ttest_ind('uts', 'A', 'a1', 'a0', ds=ds, tail=1, samples=1)
    # persistence
    string = pickle.dumps(res, pickle.HIGHEST_PROTOCOL)
    res_ = pickle.loads(string)
    assert_equal(repr(res_), repr(res))
    assert_dataobj_equal(res.p_uncorrected, res_.p_uncorrected)


def test_ttest_rel():
    "Test testnd.ttest_rel()"
    ds = datasets.get_rand()

    # basic
    res = testnd.ttest_rel('uts', 'A%B', ('a1', 'b1'), ('a0', 'b0'), 'rm',
                           ds=ds)
    repr(res)

    # persistence
    string = pickle.dumps(res, pickle.HIGHEST_PROTOCOL)
    res_ = pickle.loads(string)
    repr(res_)
    assert_equal(repr(res_), repr(res))
    assert_dataobj_equal(res.p_uncorrected, res_.p_uncorrected)

    # collapsing cells
    res2 = testnd.ttest_rel('uts', 'A', 'a1', 'a0', 'rm', ds=ds)
    assert_less(res2.p_uncorrected.min(), 0.05)
    assert_equal(res2.n, res.n)
