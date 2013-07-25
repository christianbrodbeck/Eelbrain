'''
Created on Jul 11, 2013

@author: christian
'''
from eelbrain.data import datasets, plot


def test_plot_butterfly():
    "Test plot.utsnd.butterfly"
    ds = datasets.get_rand(utsnd=True)
    p = plot.utsnd.butterfly('utsnd', ds=ds)
    p.close()
    p = plot.utsnd.butterfly('utsnd', 'A%B', ds=ds)
    p.close()

def test_plot_array():
    "Test plot.utsnd.array"
    ds = datasets.get_rand(utsnd=True)
    p = plot.utsnd.array('utsnd', 'A%B', ds=ds)
    p.close()
    p = plot.utsnd.array('utsnd', ds=ds)
    p.close()
