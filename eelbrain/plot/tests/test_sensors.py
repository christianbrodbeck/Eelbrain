'''
Created on Jul 11, 2013

@author: christian
'''
import numpy as np
from numpy.testing import assert_array_equal

from eelbrain.vessels import datasets
from eelbrain import plot


def test_multi():
    "Test plot.sensors.multi"
    ds = datasets.get_rand(utsnd=True)
    Y = ds['utsnd']
    p = plot.sensors.multi(Y)
    ROI = [1, 2]
    p.set_ROI(ROI)
    ROI2 = p.get_ROI()

    test_range = np.arange(3)
    assert_array_equal(test_range[ROI2], test_range[ROI], "ROI changed after "
                       "set/get")
