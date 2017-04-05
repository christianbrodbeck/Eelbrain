# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from nose.tools import eq_
from numpy.testing import assert_array_equal

from eelbrain import (
    NDVar, UTS, datasets, concatenate, cross_correlation, find_intervals)


def test_concatenate():
    "Test concatenate()"
    ds = datasets.get_uts(True)

    v0 = ds[0, 'utsnd']
    v1 = ds[1, 'utsnd']
    vc = concatenate((v1, v0))
    assert_array_equal(vc.sub(time=(0, 1)).x, v1.x)
    assert_array_equal(vc.sub(time=(1, 2)).x, v0.x)
    assert_array_equal(vc.info, ds['utsnd'].info)


def test_cross_correlation():
    ds = datasets._get_continuous()
    x = ds['x1']

    eq_(cross_correlation(x, x).argmax(), 0)
    eq_(cross_correlation(x[2:], x).argmax(), 0)
    eq_(cross_correlation(x[:9], x).argmax(), 0)
    eq_(cross_correlation(x, x[1:]).argmax(), 0)
    eq_(cross_correlation(x, x[:8]).argmax(), 0)
    eq_(cross_correlation(x[2:], x[:8]).argmax(), 0)


def test_find_intervals():
    time = UTS(-5, 1, 10)
    x = NDVar([0, 1, 0, 1, 1, 0, 1, 1, 1, 0], (time,))
    eq_(find_intervals(x), ((-4, -3), (-2, 0), (1, 4)))
    x = NDVar([0, 1, 0, 1, 1, 0, 1, 1, 1, 1], (time,))
    eq_(find_intervals(x), ((-4, -3), (-2, 0), (1, 5)))
    x = NDVar([1, 1, 0, 1, 1, 0, 1, 1, 1, 1], (time,))
    eq_(find_intervals(x), ((-5, -3), (-2, 0), (1, 5)))
