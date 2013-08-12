'''
Created on Jul 11, 2013

@author: christian
'''
import numpy as np
from numpy.testing import assert_array_equal

from eelbrain.data import datasets, plot


def test_map2d():
    "Test plot.SensorMap2d"
    plot.configure_backend(False, False)
    ds = datasets.get_rand(utsnd=True)
    Y = ds['utsnd']
    p = plot.SensorMap2d(Y)
    p.set_label_color('g')
    p.set_label_text('idx')
    p.show_connectivity()
    p.show_connectivity(None)
    p.close()


def test_multi():
    "Test plot.SensorMaps"
    plot.configure_backend(False, False)
    ds = datasets.get_rand(utsnd=True)
    Y = ds['utsnd']
    p = plot.SensorMaps(Y)
    ROI = [1, 2]
    p.set_ROI(ROI)
    ROI2 = p.get_ROI()

    test_range = np.arange(3)
    assert_array_equal(test_range[ROI2], test_range[ROI], "ROI changed after "
                       "set/get")

    p.close()
