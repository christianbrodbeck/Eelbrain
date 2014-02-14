# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>

from nose.tools import assert_equal, assert_in, assert_less, assert_not_in
from numpy.testing import assert_array_equal

from eelbrain.data import datasets, testnd, plot


def test_anova():
    "Test testnd.anova()"
    plot.configure_backend(False, False)
    ds = datasets.get_rand(True)

    testnd.anova('utsnd', 'A*B', ds=ds)

    res = testnd.anova('utsnd', 'A*B*rm', ds=ds)
    p = plot.Array(res)
    p.close()

    res = testnd.anova('utsnd', 'A*B*rm', ds=ds, samples=2)
    p = plot.Array(res)
    p.close()


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
    p = plot.Array(res)
    p.close()

    res = testnd.corr('utsnd', 'Y', 'rm', ds=ds, samples=2)
    p = plot.Array(res)
    p.close()


def test_ttest_1samp():
    "Test testnd.ttest_1samp()"
    ds = datasets.get_rand()

    # no clusters
    res0 = testnd.ttest_1samp('uts', sub="A == 'a0'", ds=ds)
    assert_less(res0.p.x.min(), 0.05)
    repr0 = repr(res0)
    assert_in('against', repr0)
    assert_not_in('clusters', repr0)

    # clusters without resampling
    res1 = testnd.ttest_1samp('uts', sub="A == 'a0'", ds=ds, samples=0,
                              pmin=0.05, tstart=0, tstop=0.6, mintime=0.05)
    assert_equal(res1.clusters.n_cases, 2)
    assert_not_in('p', res1.clusters)
    repr1 = repr(res1)
    assert_in('clusters', repr1)
    assert_not_in('samples', repr1)


    # clusters with resampling
    res2 = testnd.ttest_1samp('uts', sub="A == 'a0'", ds=ds, samples=10,
                              pmin=0.05, tstart=0, tstop=0.6, mintime=0.05)
    assert_equal(res2.clusters.n_cases, 2)
    assert_equal(res2._n_samples, 10)
    assert_in('p', res2.clusters)
    repr2 = repr(res2)
    assert_in('samples', repr2)
    assert_not_in('permutations', repr2)

    # clusters with permutations
    dss = ds.sub("logical_and(A=='a0', B=='b0')")[:8]
    res3 = testnd.ttest_1samp('uts', sub="A == 'a0'", ds=dss, samples=10000,
                              pmin=0.05, tstart=0, tstop=0.6, mintime=0.05)
    assert_equal(res3.clusters.n_cases, 2)
    assert_equal(res3._n_samples, 255)
    assert_less(res3.clusters['p'].x.min(), 0.05)
    repr3 = repr(res3)
    assert_in('permutations', repr3)
    assert_not_in('samples', repr3)
