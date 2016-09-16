# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from math import floor
import os

from nose.tools import eq_, assert_almost_equal
from numpy.testing import assert_array_equal, assert_allclose
import scipy.io

from eelbrain import datasets
from eelbrain._stats.boosting import boosting, boost_1seg, corr_for_kernel


def test_boosting():
    "Test boosting NDVars"
    ds = datasets._get_continuous()

    # test values from running function, not verified independently
    res = boosting(ds['y'], ds['x1'], 0, 1)
    eq_(round(res.corr, 2), 0.83)

    res = boosting(ds['y'], ds['x2'], 0, 1)
    eq_(round(res.corr, 2), 0.57)

    res = boosting(ds['y'], [ds['x1'], ds['x2']], 0, 1)
    eq_(round(res.corr, 2), 0.97)


def test_boosting_func():
    "Test boosting() against svdboostV4pred.m"
    # 1d-TRF
    path = os.path.join(os.path.dirname(__file__), 'test_boosting.mat')
    mat = scipy.io.loadmat(path)
    x = mat['stim']
    y = mat['signal'][0]

    h, test_sse_history, msg = boost_1seg(x, y, 10, 0.005, 10000, 40, 0, 0.01,
                                          'SS')
    test_seg_len = int(floor(x.shape[1] / 40))
    r, rr = corr_for_kernel(y[:test_seg_len], x[:, :test_seg_len], h, out='both')

    assert_array_equal(h, mat['h'])
    assert_almost_equal(r, mat['crlt'][0, 0], 10)
    assert_almost_equal(rr, mat['crlt'][1, 0], 10)
    assert_allclose(test_sse_history, mat['Str_testE'][0])

    # 2d-TRF
    path = os.path.join(os.path.dirname(__file__), 'test_boosting_2d.mat')
    mat = scipy.io.loadmat(path)
    x = mat['stim']
    y = mat['signal'][0]

    h, test_sse_history, msg = boost_1seg(x, y, 10, 0.005, 10000, 40, 0, 0.01,
                                          'SS')
    test_seg_len = int(floor(x.shape[1] / 40))
    r, rr = corr_for_kernel(y[:test_seg_len], x[:, :test_seg_len], h, out='both')

    assert_array_equal(h, mat['h'])
    assert_almost_equal(r, mat['crlt'][0, 0], 10)
    assert_almost_equal(rr, mat['crlt'][1, 0])
    # svdboostV4pred multiplies error by number of predictors
    assert_allclose(test_sse_history, mat['Str_testE'][0] / 3)
