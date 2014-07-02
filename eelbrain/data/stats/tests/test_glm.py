from itertools import izip

import numpy as np
from numpy import newaxis

from nose.tools import assert_equal, assert_almost_equal, assert_raises, nottest
from numpy.testing import assert_allclose

from eelbrain.data import Dataset, datasets
from eelbrain.data.test import anova
from eelbrain.data.stats.glm import _nd_anova


@nottest
def assert_f_test_equal(f_test, r_res, r_row, f=None):
    assert_equal(f_test.df, r_res[0][r_row])
    assert_almost_equal(f_test.SS, r_res[1][r_row])
    assert_almost_equal(f_test.MS, r_res[2][r_row])
    assert_almost_equal(f_test.F, r_res[3][r_row])
    assert_almost_equal(f_test.p, r_res[4][r_row])
    if f is not None:
        assert_almost_equal(f, r_res[3][r_row])


def run_on_lm_fitter(y, x, ds):
    y = ds.eval(y)
    y = np.hstack((y.x[:, newaxis], y.x[:, newaxis]))
    x = ds.eval(x)
    fitter = _nd_anova(x)
    fmaps = fitter.map(y)
    fs = fmaps[:, 0]
    return fs


def test_anova():
    "Test univariate ANOVA"
    ds = datasets.get_rand()
    aov = anova('Y', 'A*B*rm', ds=ds)
    print aov

    # not fully specified model with random effects
    assert_raises(NotImplementedError, anova, 'Y', 'A*rm', ds=ds)


def test_anova_r_sleep():
    "Test ANOVA accuracy by comparing with R (sleep dataset)"
    from rpy2.robjects import r

    # "sleep" dataset
    print r('data(sleep)')
    ds = Dataset.from_r('sleep')
    ds['ID'].random = True

    # independent measures
    aov = anova('extra', 'group', ds=ds)
    fs = run_on_lm_fitter('extra', 'group', ds)
    print r('sleep.aov <- aov(extra ~ group, sleep)')
    print r('sleep.summary <- summary(sleep.aov)')
    r_res = r['sleep.summary'][0]
    assert_f_test_equal(aov.f_tests[0], r_res, 0, fs[0])

    # repeated measures
    aov = anova('extra', 'group * ID', ds=ds)
    fs = run_on_lm_fitter('extra', 'group * ID', ds)
    print r('sleep.aov <- aov(extra ~ group + Error(ID / group), sleep)')
    print r('sleep.summary <- summary(sleep.aov)')
    r_res = r['sleep.summary'][1][0]
    assert_f_test_equal(aov.f_tests[0], r_res, 0, fs[0])

    # unbalanced (independent measures)
    ds2 = ds[1:]
    print r('sleep2 <- subset(sleep, (group == 2) | (ID != 1))')
    aov = anova('extra', 'group', ds=ds2)
#     fs = run_on_lm_fitter('extra', 'group', ds2)
    print r('sleep2.aov <- aov(extra ~ group, sleep2)')
    print r('sleep2.summary <- summary(sleep2.aov)')
    r_res = r['sleep2.summary'][0]
    assert_f_test_equal(aov.f_tests[0], r_res, 0)


def test_lmfitter():
    "Test the _nd_anova class"
    ds = datasets.get_rand()

    # independent, residuals vs. Hopkins
    y = ds['uts'].x
    y_shape = y.shape

    x = ds.eval("A * B")
    lm = _nd_anova(x)
    f_maps = lm.map(y)
    p_maps = lm.p_maps(f_maps)

    x_full = ds.eval("A * B + ind(A%B)")
    lm_full = _nd_anova(x_full)
    f_maps_full = lm_full.map(y)
    p_maps_full = lm_full.p_maps(f_maps)

    for f, f_full in izip(f_maps, f_maps_full):
        assert_allclose(f, f_full)
    for p, p_full in izip(p_maps, p_maps_full):
        assert_allclose(p, p_full)

    # repeated measures
    x = ds.eval("A * B * rm")
    lm = _nd_anova(x)
    f_maps = lm.map(y)
    p_maps = lm.p_maps(f_maps)

    aov = anova(y[:, 0], x)
    for f_test, f_map, p_map in izip(aov.f_tests, f_maps, p_maps):
        assert_almost_equal(f_map[0], f_test.F)
        assert_almost_equal(p_map[0], f_test.p)
