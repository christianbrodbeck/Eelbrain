'''
Created on Jul 11, 2013

@author: christian
'''
from eelbrain.vessels import datasets
from eelbrain import plot


def test_plot_butterfly():
    "Test plot.utsnd.butterfly"
    ds = datasets.get_rand(utsnd=True)
    plot.utsnd.butterfly('utsnd', ds=ds)
    plot.utsnd.butterfly('utsnd', 'A%B', ds=ds)

def test_plot_array():
    "Test plot.utsnd.array"
    ds = datasets.get_rand(utsnd=True)
    plot.utsnd.array('utsnd', ds=ds)
    plot.utsnd.array('utsnd', 'A%B', ds=ds)
