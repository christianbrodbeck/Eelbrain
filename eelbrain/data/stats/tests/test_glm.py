from itertools import izip

from nose.tools import assert_almost_equal, assert_raises
from numpy.testing import assert_allclose

from eelbrain.data import datasets
from eelbrain.data.test import anova
from eelbrain.data.stats.glm import LMFitter


def test_anova():
    "Test univariate ANOVA"
    ds = datasets.get_rand()
    aov = anova('Y', 'A*B*rm', ds=ds)
    print aov

    # not fully specified model with random effects
    assert_raises(NotImplementedError, anova, 'Y', 'A*rm', ds=ds)


def test_lmfitter():
    "Test the LMFitter class"
    ds = datasets.get_rand()

    # independent, residuals vs. Hopkins
    y = ds['uts'].x
    y_shape = y.shape

    x = ds.eval("A * B")
    lm = LMFitter(x, y_shape)
    f_maps = lm.map(y)
    p_maps = lm.p_maps(f_maps)

    x_full = ds.eval("A * B + ind(A%B)")
    lm_full = LMFitter(x_full, y_shape)
    f_maps_full = lm_full.map(y)
    p_maps_full = lm_full.p_maps(f_maps)

    for f, f_full in izip(f_maps, f_maps_full):
        assert_allclose(f, f_full)
    for p, p_full in izip(p_maps, p_maps_full):
        assert_allclose(p, p_full)

    # repeated measures
    x = ds.eval("A * B * rm")
    lm = LMFitter(x, y_shape)
    f_maps = lm.map(y)
    p_maps = lm.p_maps(f_maps)

    aov = anova(y[:, 0], x)
    for f_test, f_map, p_map in izip(aov.f_tests, f_maps, p_maps):
        assert_almost_equal(f_map[0], f_test.F)
        assert_almost_equal(p_map[0], f_test.p)
