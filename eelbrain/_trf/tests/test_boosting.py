# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from math import floor
import os

from nose.tools import eq_, assert_almost_equal, assert_is_none
from numpy.testing import assert_array_equal, assert_allclose
import cPickle as pickle
import scipy.io
from eelbrain import boosting, datasets
from eelbrain._trf._boosting import boost_1seg, evaluate_kernel


def assert_res_equal(res1, res):
    assert_array_equal(res1.h, res.h)
    eq_(res1.r, res.r)
    eq_(res1.spearmanr, res.spearmanr)


def test_boosting():
    "Test boosting NDVars"
    ds = datasets._get_continuous()
    y = ds['y']
    x1 = ds['x1']
    x2 = ds['x2']
    y_mean = y.mean()
    x2_mean = x2.mean()

    # test values from running function, not verified independently
    res = boosting(y, x1 * 2000, 0, 1, scale_data=False)
    eq_(repr(res), '<boosting y ~ x1, 0 - 1, scale_data=False>')
    eq_(round(res.r, 2), 0.75)
    assert_is_none(res.y_mean)

    res = boosting(y, x1, 0, 1)
    eq_(repr(res), '<boosting y ~ x1, 0 - 1>')
    eq_(round(res.r, 2), 0.83)
    eq_(res.y_mean, y_mean)
    eq_(res.y_scale, y.std())
    eq_(res.x_mean, x1.mean())
    eq_(res.x_scale, x1.std())
    # inplace
    res_ip = boosting(y.copy(), x1.copy(), 0, 1, 'inplace')
    assert_res_equal(res_ip, res)
    # persistence
    res_p = pickle.loads(pickle.dumps(res, pickle.HIGHEST_PROTOCOL))
    assert_res_equal(res_p, res)

    res = boosting(y, x2, 0, 1)
    eq_(round(res.r, 2), 0.60)

    res = boosting(y, x2, 0, 1, error='l1')
    eq_(round(res.r, 2), 0.55)
    eq_(res.y_mean, y.mean())
    eq_(res.y_scale, (y - y_mean).abs().mean())
    eq_(res.x_mean, x2_mean)
    eq_(res.x_scale, (x2 - x2_mean).abs().mean())

    res = boosting(y, [x1, x2], 0, 1)
    eq_(round(res.r, 2), 0.98)


def test_boosting_func():
    "Test boosting() against svdboostV4pred.m"
    # 1d-TRF
    path = os.path.join(os.path.dirname(__file__), 'test_boosting.mat')
    mat = scipy.io.loadmat(path)
    x = mat['stim']
    y = mat['signal'][0]

    h, test_sse_history, msg = boost_1seg(x, y, 10, 0.005, 40, 0, 0.01, 'l2')
    test_seg_len = int(floor(x.shape[1] / 40))
    r, rr, _ = evaluate_kernel(y[:test_seg_len], x[:, :test_seg_len], h, 'l2')

    assert_array_equal(h, mat['h'])
    assert_almost_equal(r, mat['crlt'][0, 0], 10)
    assert_almost_equal(rr, mat['crlt'][1, 0], 10)
    assert_allclose(test_sse_history, mat['Str_testE'][0])

    # 2d-TRF
    path = os.path.join(os.path.dirname(__file__), 'test_boosting_2d.mat')
    mat = scipy.io.loadmat(path)
    x = mat['stim']
    y = mat['signal'][0]

    h, test_sse_history, msg = boost_1seg(x, y, 10, 0.005, 40, 0, 0.01, 'l2')
    test_seg_len = int(floor(x.shape[1] / 40))
    r, rr, _ = evaluate_kernel(y[:test_seg_len], x[:, :test_seg_len], h, 'l2')

    assert_array_equal(h, mat['h'])
    assert_almost_equal(r, mat['crlt'][0, 0], 10)
    assert_almost_equal(rr, mat['crlt'][1, 0])
    # svdboostV4pred multiplies error by number of predictors
    assert_allclose(test_sse_history, mat['Str_testE'][0] / 3)
