# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>

from itertools import izip, product
import cPickle as pickle
import logging

from nose.tools import (eq_, assert_equal, assert_greater_equal, assert_less,
                        assert_in, assert_not_in, assert_raises)
import numpy as np
from numpy.testing import assert_array_equal
from scipy import ndimage

import eelbrain
from eelbrain import datasets, testnd, NDVar
from eelbrain._data_obj import UTS, Ordered, Sensor, cwt_morlet
from eelbrain._stats import testnd as _testnd
from eelbrain._stats.testnd import _ClusterDist, label_clusters
from eelbrain._utils import logger
from eelbrain.tests.test_data import assert_dataobj_equal, assert_dataset_equal


def test_anova():
    "Test testnd.anova()"
    ds = datasets.get_uts(True)

    testnd.anova('utsnd', 'A*B', ds=ds)
    for samples in (0, 2):
        logger.info("TEST:  samples=%r" % samples)
        testnd.anova('utsnd', 'A*B', ds=ds, samples=samples)
        testnd.anova('utsnd', 'A*B', ds=ds, samples=samples, pmin=0.05)
        testnd.anova('utsnd', 'A*B', ds=ds, samples=samples, tfce=True)

    res = testnd.anova('utsnd', 'A*B*rm', ds=ds, samples=0, pmin=0.05)
    repr(res)
    res = testnd.anova('utsnd', 'A*B*rm', ds=ds, samples=2, pmin=0.05)
    repr(res)

    # persistence
    string = pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL)
    res_ = pickle.loads(string)
    assert_equal(repr(res_), repr(res))

    # threshold-free
    res = testnd.anova('utsnd', 'A*B*rm', ds=ds, samples=10)
    repr(res)
    assert_in('A clusters', res.clusters.info)
    assert_in('B clusters', res.clusters.info)
    assert_in('A x B clusters', res.clusters.info)

    # no clusters
    res = testnd.anova('uts', 'B', sub="A=='a1'", ds=ds, samples=5, pmin=0.05,
                       mintime=0.02)
    repr(res)
    assert_in('v', res.clusters)
    assert_in('p', res.clusters)

    # all effects with clusters
    res = testnd.anova('uts', 'A*B*rm', ds=ds, samples=5, pmin=0.05,
                       tstart=0.1, mintime=0.02)
    assert_equal(set(res.clusters['effect'].cells), set(res.effects))

    # some effects with clusters, some without
    res = testnd.anova('uts', 'A*B*rm', ds=ds, samples=5, pmin=0.05,
                       tstart=0.37, mintime=0.02)
    string = pickle.dumps(res, pickle.HIGHEST_PROTOCOL)
    res_ = pickle.loads(string)
    assert_dataobj_equal(res.clusters, res_.clusters)

    # test multi-effect results (with persistence)
    # UTS
    res = testnd.anova('uts', 'A*B*rm', ds=ds, samples=5)
    repr(res)
    string = pickle.dumps(res, pickle.HIGHEST_PROTOCOL)
    resr = pickle.loads(string)
    tf_clusters = resr.find_clusters(pmin=0.05)
    peaks = resr.find_peaks()
    assert_dataobj_equal(tf_clusters, res.find_clusters(pmin=0.05))
    assert_dataobj_equal(peaks, res.find_peaks())
    assert_equal(tf_clusters.eval("p.min()"), peaks.eval("p.min()"))
    unmasked = resr.f[0]
    masked = resr.masked_parameter_map(effect=0, pmin=0.05)
    assert_array_equal(masked.x <= unmasked.x, True)

    # reproducibility
    res0 = testnd.anova('utsnd', 'A*B*rm', ds=ds, pmin=0.05, samples=5)
    res = testnd.anova('utsnd', 'A*B*rm', ds=ds, pmin=0.05, samples=5)
    assert_dataset_equal(res.clusters, res0.clusters)
    eelbrain._stats.testnd.MULTIPROCESSING = 0
    res = testnd.anova('utsnd', 'A*B*rm', ds=ds, pmin=0.05, samples=5)
    assert_dataset_equal(res.clusters, res0.clusters)
    eelbrain._stats.testnd.MULTIPROCESSING = 1

    # permutation
    eelbrain._stats.permutation._YIELD_ORIGINAL = 1
    samples = 4
    # raw
    res = testnd.anova('utsnd', 'A*B*rm', ds=ds, samples=samples)
    for dist in res._cdist:
        eq_(len(dist.dist), samples)
        assert_array_equal(dist.dist, dist.parameter_map.abs().max())
    # TFCE
    res = testnd.anova('utsnd', 'A*B*rm', ds=ds, tfce=True, samples=samples)
    for dist in res._cdist:
        eq_(len(dist.dist), samples)
        assert_array_equal(dist.dist, dist.tfce_map.abs().max())
    # thresholded
    res = testnd.anova('utsnd', 'A*B*rm', ds=ds, pmin=0.05, samples=samples)
    clusters = res.find_clusters()
    for dist, effect in izip(res._cdist, res.effects):
        effect_idx = clusters.eval("effect == %r" % effect)
        vmax = clusters[effect_idx, 'v'].abs().max()
        eq_(len(dist.dist), samples)
        assert_array_equal(dist.dist, vmax)
    eelbrain._stats.permutation._YIELD_ORIGINAL = 0


def test_anova_incremental():
    "Test testnd.anova() with incremental f-tests"
    ds = datasets.get_uts()
    testnd.anova('uts', 'A*B', ds=ds[3:], pmin=0.05, samples=10)


def test_clusterdist():
    "Test _ClusterDist class"
    shape = (10, 6, 6, 4)
    locs = [[0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0]]
    x = np.random.normal(0, 1, shape)
    sensor = Sensor(locs, ['0', '1', '2', '3'])
    sensor.set_connectivity(connect_dist=1.1)
    dims = ('case', UTS(-0.1, 0.1, 6), Ordered('dim2', range(6), 'unit'),
            sensor)
    y = NDVar(x, dims)

    # test connecting sensors
    logger.info("TEST:  connecting sensors")
    bin_map = np.zeros(shape[1:], dtype=np.bool8)
    bin_map[:3, :3, :2] = True
    pmap = np.random.normal(0, 1, shape[1:])
    np.clip(pmap, -1, 1, pmap)
    pmap[bin_map] = 2
    cdist = _ClusterDist(y, 0, 1.5)
    print repr(cdist)
    cdist.add_original(pmap)
    print repr(cdist)
    assert_equal(cdist.n_clusters, 1)
    assert_array_equal(cdist._original_cluster_map == cdist._cids[0],
                       cdist._crop(bin_map).swapaxes(0, cdist._nad_ax))
    assert_equal(cdist.parameter_map.dims, y.dims[1:])

    # test connecting many sensors
    logger.info("TEST:  connecting sensors")
    bin_map = np.zeros(shape[1:], dtype=np.bool8)
    bin_map[:3, :3] = True
    pmap = np.random.normal(0, 1, shape[1:])
    np.clip(pmap, -1, 1, pmap)
    pmap[bin_map] = 2
    cdist = _ClusterDist(y, 0, 1.5)
    cdist.add_original(pmap)
    assert_equal(cdist.n_clusters, 1)
    assert_array_equal(cdist._original_cluster_map == cdist._cids[0],
                       cdist._crop(bin_map).swapaxes(0, cdist._nad_ax))

    # test keeping sensors separate
    logger.info("TEST:  keeping sensors separate")
    bin_map = np.zeros(shape[1:], dtype=np.bool8)
    bin_map[:3, :3, 0] = True
    bin_map[:3, :3, 2] = True
    pmap = np.random.normal(0, 1, shape[1:])
    np.clip(pmap, -1, 1, pmap)
    pmap[bin_map] = 2
    cdist = _ClusterDist(y, 1, 1.5)
    cdist.add_original(pmap)
    assert_equal(cdist.n_clusters, 2)

    # criteria
    ds = datasets.get_uts(True)
    res = testnd.ttest_rel('utsnd', 'A', match='rm', ds=ds, samples=0, pmin=0.05)
    assert_less(res.clusters['duration'].min(), 0.01)
    eq_(res.clusters['n_sensors'].min(), 1)
    res = testnd.ttest_rel('utsnd', 'A', match='rm', ds=ds, samples=0, pmin=0.05,
                           mintime=0.02, minsensor=2)
    assert_greater_equal(res.clusters['duration'].min(), 0.02)
    eq_(res.clusters['n_sensors'].min(), 2)

    # 1d
    res1d = testnd.ttest_rel('utsnd.sub(time=0.1)', 'A', match='rm', ds=ds,
                             samples=0, pmin=0.05)
    assert_dataobj_equal(res1d.p_uncorrected, res.p_uncorrected.sub(time=0.1))

    # TFCE
    logger.info("TEST:  TFCE")
    sensor = Sensor(locs, ['0', '1', '2', '3'])
    sensor.set_connectivity(connect_dist=1.1)
    dims = ('case', UTS(-0.1, 0.1, 4), sensor,
            Ordered('dim2', range(10), 'unit'))
    y = NDVar(np.random.normal(0, 1, (10, 4, 4, 10)), dims)
    cdist = _ClusterDist(y, 3, None)
    cdist.add_original(y.x[0])
    cdist.finalize()
    assert_equal(cdist.dist.shape, (3,))
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

    mps = False, True
    thresholds = (None, 'tfce')
    for mp, threshold in product(mps, thresholds):
        logger.info("TEST:  multiprocessing=%r, threshold=%r" % (mp, threshold))
        _testnd.multiprocessing = mp

        # test keeping dimension
        cdist = _ClusterDist(y, 5, threshold, dist_dim='sensor')
        print repr(cdist)
        cdist.add_original(y.x[0])
        print repr(cdist)
        assert_equal(cdist.dist.shape, (5, 4))

        # test keeping time bins
        cdist = _ClusterDist(y, 5, threshold, dist_tstep=0.2)
        cdist.add_original(y.x[0])
        assert_equal(cdist.dist.shape, (5, 2))
        assert_raises(ValueError, _ClusterDist, y, 5, threshold, dist_tstep=0.3)

        # test keeping dimension and time bins
        cdist = _ClusterDist(y, 5, threshold, dist_dim='sensor', dist_tstep=0.2)
        cdist.add_original(y.x[0])
        assert_equal(cdist.dist.shape, (5, 4, 2))

        # test keeping 2 dimensions and time bins
        cdist = _ClusterDist(y, 5, threshold, dist_dim=('sensor', 'dim2'),
                             dist_tstep=0.2)
        cdist.add_original(y.x[0])
        assert_equal(cdist.dist.shape, (5, 4, 2, 10))


def test_corr():
    "Test testnd.corr()"
    ds = datasets.get_uts(True)

    # add correlation
    Y = ds['Y']
    utsnd = ds['utsnd']
    utsnd.x[:, 3:5, 50:65] += Y.x[:, None, None]

    res = testnd.corr('utsnd', 'Y', 'rm', ds=ds)
    repr(res)
    res = testnd.corr('utsnd', 'Y', 'rm', ds=ds, samples=10, pmin=0.05)
    repr(res)

    # persistence
    string = pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL)
    res_ = pickle.loads(string)
    assert_equal(repr(res_), repr(res))
    assert_dataobj_equal(res.p_uncorrected, res_.p_uncorrected)
    assert_dataobj_equal(res.p, res_.p)


def test_t_contrast():
    ds = datasets.get_uts()

    # simple contrast
    res = testnd.t_contrast_rel('uts', 'A', 'a1>a0', 'rm', ds=ds, samples=10,
                                pmin=0.05)
    repr(res)
    res_ = testnd.ttest_rel('uts', 'A', 'a1', 'a0', 'rm', ds=ds)
    assert_array_equal(res.t.x, res_.t.x)
    assert_in('samples', repr(res))

    # complex contrast
    res = testnd.t_contrast_rel('uts', 'A%B', 'min(a0|b0>a1|b0, a0|b1>a1|b1)',
                                'rm', ds=ds, samples=10, pmin=0.05)
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

    # contrast with "*"
    res = testnd.t_contrast_rel('uts', 'A%B', 'min(a1|b0>a0|*, a1|b1>a0|*)',
                                'rm', ds=ds, tail=1)


def test_labeling():
    "Test cluster labeling"
    shape = flat_shape = (4, 20)
    pmap = np.empty(shape, np.float_)
    struct = ndimage.generate_binary_structure(2, 1)
    struct[::2] = False
    conn = np.array([(0, 1), (0, 3), (1, 2), (2, 3)], np.uint32)
    criteria = None

    # some clusters
    pmap[:] = [[ 3, 3, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0],
               [ 0, 1, 0, 0, 0, 0, 8, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0, 0, 4, 0],
               [ 0, 3, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 4, 4],
               [ 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0]]
    cmap, cids = label_clusters(pmap, 2, 0, conn, criteria)
    assert_equal(len(cids), 6)
    assert_array_equal(cmap > 0, np.abs(pmap) > 2)

    # some other clusters
    pmap[:] = [[ 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0],
               [ 0, 4, 0, 0, 0, 0, 0, 4, 0, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0],
               [ 0, 0, 4, 4, 0, 4, 4, 0, 4, 0, 0, 0, 4, 4, 1, 0, 4, 4, 0, 0],
               [ 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0]]
    cmap, cids = label_clusters(pmap, 2, 0, conn, criteria)
    assert_equal(len(cids), 6)
    assert_array_equal(cmap > 0, np.abs(pmap) > 2)


def test_ttest_1samp():
    "Test testnd.ttest_1samp()"
    ds = datasets.get_uts(True)

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
    assert_equal(res1.clusters.n_cases, 1)
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
    assert_equal(res2.clusters.n_cases, 1)
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

    # nd
    dss = ds.sub("A == 'a0'")
    res = testnd.ttest_1samp('utsnd', ds=dss, samples=1)
    res = testnd.ttest_1samp('utsnd', ds=dss, pmin=0.05, samples=1)
    res = testnd.ttest_1samp('utsnd', ds=dss, tfce=True, samples=1)

    # TFCE properties
    res = testnd.ttest_1samp('utsnd', sub="A == 'a0'", ds=ds, samples=1)
    string = pickle.dumps(res, pickle.HIGHEST_PROTOCOL)
    res = pickle.loads(string)
    tfce_clusters = res.find_clusters(pmin=0.05)
    peaks = res.find_peaks()
    assert_equal(tfce_clusters.eval("p.min()"), peaks.eval("p.min()"))
    masked = res.masked_parameter_map(pmin=0.05)
    assert_array_equal(masked.abs().x <= res.t.abs().x, True)


def test_ttest_ind():
    "Test testnd.ttest_ind()"
    ds = datasets.get_uts(True)

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

    # nd
    res = testnd.ttest_ind('utsnd', 'A', 'a1', 'a0', ds=ds, pmin=0.05, samples=2)
    eq_(res._cdist.n_clusters, 10)


def test_ttest_rel():
    "Test testnd.ttest_rel()"
    ds = datasets.get_uts(True)

    # basic
    res = testnd.ttest_rel('uts', 'A%B', ('a1', 'b1'), ('a0', 'b0'), 'rm',
                           ds=ds, samples=100)
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

    # reproducibility
    res3 = testnd.ttest_rel('uts', 'A%B', ('a1', 'b1'), ('a0', 'b0'), 'rm',
                            ds=ds, samples=100)
    assert_dataset_equal(res3.find_clusters(maps=True), res.clusters)
    eelbrain._stats.testnd.MULTIPROCESSING = 0
    res4 = testnd.ttest_rel('uts', 'A%B', ('a1', 'b1'), ('a0', 'b0'), 'rm',
                            ds=ds, samples=100)
    assert_dataset_equal(res4.find_clusters(maps=True), res.clusters)
    eelbrain._stats.testnd.MULTIPROCESSING = 1
    sds = ds.sub("B=='b0'")
    # thresholded, UTS
    eelbrain._stats.testnd.MULTIPROCESSING = 0
    res0 = testnd.ttest_rel('uts', 'A', 'a1', 'a0', 'rm', ds=sds, pmin=0.1,
                            samples=100)
    tgt = res0.find_clusters()
    eelbrain._stats.testnd.MULTIPROCESSING = 1
    res1 = testnd.ttest_rel('uts', 'A', 'a1', 'a0', 'rm', ds=sds, pmin=0.1,
                            samples=100)
    assert_dataset_equal(res1.find_clusters(), tgt)
    # thresholded, UTSND
    eelbrain._stats.testnd.MULTIPROCESSING = 0
    res0 = testnd.ttest_rel('utsnd', 'A', 'a1', 'a0', 'rm', ds=sds, pmin=0.1,
                            samples=100)
    tgt = res0.find_clusters()
    eelbrain._stats.testnd.MULTIPROCESSING = 1
    res1 = testnd.ttest_rel('utsnd', 'A', 'a1', 'a0', 'rm', ds=sds, pmin=0.1,
                            samples=100)
    assert_dataset_equal(res1.find_clusters(), tgt)
    # TFCE, UTS
    eelbrain._stats.testnd.MULTIPROCESSING = 0
    res0 = testnd.ttest_rel('uts', 'A', 'a1', 'a0', 'rm', ds=sds, tfce=True,
                            samples=10)
    tgt = res0.compute_probability_map()
    eelbrain._stats.testnd.MULTIPROCESSING = 1
    res1 = testnd.ttest_rel('uts', 'A', 'a1', 'a0', 'rm', ds=sds, tfce=True,
                            samples=10)
    assert_dataobj_equal(res1.compute_probability_map(), tgt)


def test_cwt():
    "Test tests with wavelet transform"
    ds = datasets.get_uts(True)
    ds['cwt'] = cwt_morlet(ds['utsnd'], np.arange(10, 20))
    res = testnd.ttest_rel('cwt', 'A', match='rm', ds=ds, pmin=0.05, samples=10)
    cluster = res.clusters.sub("p == 0")
    assert_array_equal(cluster['frequency_min'], 10)
    assert_array_equal(cluster['frequency_max'], 19)
