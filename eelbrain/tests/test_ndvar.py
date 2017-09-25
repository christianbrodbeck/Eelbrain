# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from nose.tools import eq_
import numpy as np
from numpy.testing import assert_array_equal

from eelbrain import (
    NDVar, Scalar, UTS, datasets,
    concatenate, convolve, cross_correlation, find_intervals, find_peaks,
    psd_welch,
)


def test_concatenate():
    """Test concatenate()

    Concatenation of SourceSpace is tested in .test_mne.test_source_estimate
    """
    ds = datasets.get_uts(True)

    v0 = ds[0, 'utsnd']
    v1 = ds[1, 'utsnd']
    vc = concatenate((v1, v0))
    assert_array_equal(vc.sub(time=(0, 1)).x, v1.x)
    assert_array_equal(vc.sub(time=(1, 2)).x, v0.x)
    assert_array_equal(vc.info, ds['utsnd'].info)

    # scalar
    psd = psd_welch(ds['utsnd'], n_fft=100)
    v0 = psd.sub(frequency=(None, 5))
    v1 = psd.sub(frequency=(45, None))
    conc = concatenate((v0, v1), 'frequency')
    assert_array_equal(conc.frequency.values[:5], psd.frequency.values[:5])
    assert_array_equal(conc.frequency.values[5:], psd.frequency.values[45:])
    conc_data = conc.get_data(v1.dimnames)
    assert_array_equal(conc_data[:, :, 5:], v1.x)


def test_convolve():
    ds = datasets._get_continuous()

    h1 = ds['h1']
    h2 = ds['h2']
    x1 = ds['x1']

    xc = convolve(h1, x1)
    xc_np = np.convolve(h1.x, x1.x)
    assert_array_equal(xc.x, xc_np[:100])

    # add dimension through kernel
    xc = convolve(h2, x1)
    xc_np = np.vstack((np.convolve(h2.x[0], x1.x)[:100],
                       np.convolve(h2.x[1], x1.x)[:100]))
    assert_array_equal(xc.x, xc_np)


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


def test_find_peaks():
    scalar = Scalar('scalar', range(9))
    time = UTS(0, .1, 12)
    v = NDVar(np.zeros((9, 12)), (scalar, time))
    wsize = [0, 0, 1, 2, 3, 2, 1, 0, 0]
    for i, s in enumerate(wsize):
        if s:
            v.x[i, 5 - s: 5 + s] += np.hamming(2 * s)

    peaks = find_peaks(v)
    x, y = np.where(peaks.x)
    assert_array_equal(x, [4])
    assert_array_equal(y, [5])
