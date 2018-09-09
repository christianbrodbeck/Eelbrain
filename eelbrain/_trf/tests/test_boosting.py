# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from math import floor
import os
from warnings import catch_warnings, filterwarnings

from nose.tools import eq_, assert_almost_equal, assert_is_none, assert_raises
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pickle
import scipy.io
from eelbrain import (
    datasets, test, configure,
    boosting, convolve, correlation_coefficient, epoch_impulse_predictor,
)

from eelbrain._utils.testing import assert_dataobj_equal
from eelbrain._trf._boosting import boost, evaluate_kernel


def assert_res_equal(res1, res):
    assert_array_equal(res1.h, res.h)
    eq_(res1.r, res.r)
    eq_(res1.spearmanr, res.spearmanr)


def run_boosting(ds):
    "Run boosting tests"
    y = ds['y']
    x1 = ds['x1']
    x2 = ds['x2']
    y_mean = y.mean()
    x2_mean = x2.mean()

    # test values from running function, not verified independently
    res = boosting(y, x1 * 2000, 0, 1, scale_data=False, mindelta=0.0025)
    eq_(repr(res),
        '<boosting y ~ x1, 0 - 1, scale_data=False, mindelta=0.0025>')
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
    eq_(round(res.r, 2), 0.95)


def test_boosting():
    "Test boosting NDVars"
    ds = datasets._get_continuous()

    # test boosting results
    configure(n_workers=0)
    yield run_boosting, ds
    configure(n_workers=True)
    yield run_boosting, ds


def test_boosting_epochs():
    """Test boosting with epoched data"""
    ds = datasets.get_uts(True)
    p1 = epoch_impulse_predictor('uts', 'A=="a1"', name='a1', ds=ds)
    p0 = epoch_impulse_predictor('uts', 'A=="a0"', name='a0', ds=ds)
    p1 = p1.smooth('time', .05, 'hamming')
    p0 = p0.smooth('time', .05, 'hamming')
    # 1d
    for tstart in (-0.1, 0.1, 0):
        print(f"tstart={tstart}")
        res = boosting('uts', [p0, p1], tstart, 0.6, model='A', ds=ds, debug=True)
        y = convolve(res.h_scaled, [p0, p1])
        assert correlation_coefficient(y, res.y_pred) > .999
        r = correlation_coefficient(y, ds['uts'])
        assert_almost_equal(res.r, r, 3)
        assert res.n_segments == 10
    assert_almost_equal(res.h[0].rms(), 0.0136, 3)
    assert_almost_equal(res.h[1].rms(), 0.000569, 3)
    # 2d
    res = boosting('utsnd', [p0, p1], 0, 0.6, model='A', ds=ds)
    eq_(len(res.h), 2)
    eq_(res.h[0].shape, (5, 60))
    eq_(res.h[1].shape, (5, 60))
    y = convolve(res.h_scaled, [p0, p1])
    r = correlation_coefficient(y, ds['utsnd'], ('case', 'time'))
    assert_dataobj_equal(res.r, r, decimal=3, name=False)


def test_result():
    "Test boosting results"
    ds = datasets._get_continuous()
    x1 = ds['x1']

    # convolve function
    y = convolve([ds['h1'], ds['h2']], [ds['x1'], ds['x2']])
    assert_dataobj_equal(y, ds['y'], name=False)

    # test prediction with res.h and res.h_scaled
    res = boosting(ds['y'], ds['x1'], 0, 1)
    y1 = convolve(res.h_scaled, ds['x1'])
    x_scaled = ds['x1'] / res.x_scale
    y2 = convolve(res.h, x_scaled)
    y2 *= res.y_scale
    y2 += y1.mean() - y2.mean()  # mean can't be reconstructed
    assert_dataobj_equal(y1, y2, decimal=12)
    # reconstruction
    res = boosting(x1, y, -1, 0, debug=True)
    x1r = convolve(res.h_scaled, y)
    assert correlation_coefficient(res.y_pred, x1r) > .999
    assert_almost_equal(correlation_coefficient(x1r[0.9:], x1[0.9:]), res.r, 3)

    # test NaN checks  (modifies data)
    ds['x2'].x[1, 50] = np.nan
    assert_raises(ValueError, boosting, ds['y'], ds['x2'], 0, .5)
    assert_raises(ValueError, boosting, ds['y'], ds['x2'], 0, .5, False)
    ds['x2'].x[1, :] = 1
    with catch_warnings():
        filterwarnings('ignore', category=RuntimeWarning)
        assert_raises(ValueError, boosting, ds['y'], ds['x2'], 0, .5)
        ds['y'].x[50] = np.nan
        assert_raises(ValueError, boosting, ds['y'], ds['x1'], 0, .5)
        assert_raises(ValueError, boosting, ds['y'], ds['x1'], 0, .5, False)


def test_boosting_func():
    "Test boosting() against svdboostV4pred.m"
    # 1d-TRF
    path = os.path.join(os.path.dirname(__file__), 'test_boosting.mat')
    mat = scipy.io.loadmat(path)
    y = mat['signal'][0]
    x = mat['stim']
    x_pads = np.zeros(len(x))

    y_len = len(y)
    seg_len = int(y_len / 40)
    all_segments = np.array([[0, seg_len], [seg_len, y_len]], np.int64)
    train_segments = all_segments[1:]
    test_segments = all_segments[:1]
    h, test_sse_history = boost(y, x, x_pads, all_segments, train_segments, test_segments,
                                0, 10, 0.005, 0.005, 'l2', True)
    test_seg_len = int(floor(x.shape[1] / 40))
    r, rr, _ = evaluate_kernel(y[:test_seg_len], x[:, :test_seg_len], x_pads, h, 0, 'l2', h.shape[1] - 1)

    assert_array_equal(h, mat['h'])
    assert_almost_equal(r, mat['crlt'][0, 0], 10)
    assert_almost_equal(rr, mat['crlt'][1, 0], 10)
    assert_allclose(test_sse_history, mat['Str_testE'][0])

    # 2d-TRF
    path = os.path.join(os.path.dirname(__file__), 'test_boosting_2d.mat')
    mat = scipy.io.loadmat(path)
    y = mat['signal'][0]
    x = mat['stim']
    x_pads = np.zeros(len(x))

    h, test_sse_history = boost(y, x, x_pads, all_segments, train_segments, test_segments,
                                0, 10, 0.005, 0.005, 'l2', True)
    test_seg_len = int(floor(x.shape[1] / 40))
    r, rr, _ = evaluate_kernel(y[:test_seg_len], x[:, :test_seg_len], x_pads, h, 0, 'l2', h.shape[1] - 1)

    assert_array_equal(h, mat['h'])
    assert_almost_equal(r, mat['crlt'][0, 0], 10)
    assert_almost_equal(rr, mat['crlt'][1, 0])
    # svdboostV4pred multiplies error by number of predictors
    assert_allclose(test_sse_history, mat['Str_testE'][0] / 3)
