# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from numpy.testing import assert_array_equal

from eelbrain import datasets, test
from eelbrain._stats import test as _test


def test_star():
    "Test the star function"
    assert_array_equal(_test.star([0.1, 0.04, 0.01], int), [0, 1, 2])
    assert_array_equal(_test.star([0.001, 0.04, 0.1], int), [3, 1, 0])


def test_ttest():
    "Test test.ttest"
    ds = datasets.get_uv()

    print test.ttest('fltvar', ds=ds)
    print test.ttest('fltvar', 'A', ds=ds)
    print test.ttest('fltvar', 'A%B', ds=ds)
    print test.ttest('fltvar', 'A', match='rm', ds=ds)
    print test.ttest('fltvar', 'A', 'a1', match='rm', ds=ds)
    print test.ttest('fltvar', 'A%B', ('a1', 'b1'), match='rm', ds=ds)
