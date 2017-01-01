# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from nose.tools import assert_almost_equal
import numpy as np

from eelbrain._stats.error_functions import l1, l2

PRECISION = 10


# numpy-based error functions
#############################
def np_l2(x):
    return np.dot(x, x[:, None])[0]


def np_l1(x):
    return np.abs(x).sum()


# test function
###############
def test_error_functions():
    "Test optimized error functions"
    x = np.random.normal(0., 1., 100)

    assert_almost_equal(l1(x), np_l1(x), PRECISION)
    assert_almost_equal(l2(x), np_l2(x), PRECISION)
