# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import os

from nose.tools import eq_, assert_almost_equal
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import scipy.io

from eelbrain._stats.boosting import boosting


def test_boosting():
    """Test boosting() against svdboostV4pred.m"""
    # 1d-TRF
    path = os.path.join(os.path.dirname(__file__), 'test_boosting.mat')
    mat = scipy.io.loadmat(path)

    h_init = np.zeros((1, 10))
    h, corr, rcorr, test_sse_history, train_corr = \
        boosting(mat['stim'], mat['signal'][0], h_init, 0.005, 10000, 0)

    assert_array_equal(h, mat['h'])
    assert_almost_equal(corr, mat['crlt'][0, 0], 10)
    eq_(rcorr, mat['crlt'][1, 0])
    assert_allclose(test_sse_history, mat['Str_testE'][0])
    # NumPy 1.11 has equal_nan=True
    assert_allclose(train_corr[1:], mat['CR_train'][0, 1:])

    # 2d-TRF
    path = os.path.join(os.path.dirname(__file__), 'test_boosting_2d.mat')
    mat = scipy.io.loadmat(path)

    h_init = np.zeros((3, 10))
    h, corr, rcorr, test_sse_history, train_corr = \
        boosting(mat['stim'], mat['signal'][0], h_init, 0.005, 10000, 0)

    assert_array_equal(h, mat['h'])
    assert_almost_equal(corr, mat['crlt'][0, 0], 10)
    assert_almost_equal(rcorr, mat['crlt'][1, 0])
    # svdboostV4pred multiplies error by number of predictors
    assert_allclose(test_sse_history, mat['Str_testE'][0] / 3)
    # NumPy 1.11 has equal_nan=True
    assert_allclose(train_corr[1:], mat['CR_train'][0, 1:])
