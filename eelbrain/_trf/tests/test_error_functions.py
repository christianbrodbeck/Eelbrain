# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from nose.tools import assert_almost_equal
import numpy as np

from eelbrain._trf._boosting_opt import l1, l2

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
    index = np.array(((0, 100),), np.int64)

    assert_almost_equal(l1(x, index), np_l1(x), PRECISION)
    assert_almost_equal(l2(x, index), np_l2(x), PRECISION)
