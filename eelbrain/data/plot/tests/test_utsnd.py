'''
Created on Jul 11, 2013

@author: christian
'''
from eelbrain.data import datasets, plot


def test_plot_butterfly():
    "Test plot.Butterfly"
    plot.configure_backend(False, False)
    ds = datasets.get_rand(utsnd=True)
    p = plot.Butterfly('utsnd', ds=ds)
    p.close()
    p = plot.Butterfly('utsnd', 'A%B', ds=ds)
    p.close()

def test_plot_array():
    "Test plot.Array"
    plot.configure_backend(False, False)
    ds = datasets.get_rand(utsnd=True)
    p = plot.Array('utsnd', 'A%B', ds=ds)
    p.close()
    p = plot.Array('utsnd', ds=ds)
    p.close()
