# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import numpy as np
import pytest

from eelbrain._trf._fit_metrics import error_for_indexes


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
    assert error_for_indexes(x, index, 1) == pytest.approx(np_l1(x))
    assert error_for_indexes(x, index, 2) == pytest.approx(np_l2(x))
