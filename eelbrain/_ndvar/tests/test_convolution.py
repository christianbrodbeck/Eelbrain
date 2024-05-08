# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import numpy
from numpy.testing import assert_equal

from eelbrain._ndvar._convolve import convolve_1d


def test_convolve_1d():
    # n_x = 1
    pad_0 = numpy.array([0], float)
    pad_1 = numpy.array([1], float)
    h = numpy.array([[1, 2]], float)
    x = numpy.array([[0, 1, 0, 0, 1.5, 0]])
    yt = numpy.array([0, 1, 2, 0, 1.5, 3])  # target
    n_times = x.shape[1]
    segments = numpy.array([[0, n_times]], numpy.int64)
    yp = numpy.empty(yt.shape)
    convolve_1d(h, x, pad_0, 0, segments, yp)
    assert_equal(yp, yt)
    # shift -1
    yt = numpy.array([1, 2, 0, 1.5, 3, 0])  # target
    convolve_1d(h, x, pad_0, -1, segments, yp)
    assert_equal(yp, yt)
    # shift +1
    yt = numpy.array([0, 0, 1, 2, 0, 1.5])  # target
    convolve_1d(h, x, pad_0, 1, segments, yp)
    assert_equal(yp, yt)
    # pad head
    yt = numpy.array([2, 1, 2, 0, 1.5, 3])  # target
    convolve_1d(h, x, pad_1, 0, segments, yp)
    assert_equal(yp, yt)
    # pad tail
    yt = numpy.array([1, 2, 0, 1.5, 3, 1])  # target
    convolve_1d(h, x, pad_1, -1, segments, yp)
    assert_equal(yp, yt)
