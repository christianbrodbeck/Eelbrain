'''
Created on Jul 22, 2013

@author: christian
'''
from eelbrain.data import datasets, plot


def test_plot_topomap():
    "Test plot.topo.topomap"
    ds = datasets.get_rand(utsnd=True)
    p = plot.topo.topomap('utsnd.summary(time=(0.075, 0.125))', ds=ds)
    p.close()
    p = plot.topo.topomap('utsnd.summary(time=(0.075, 0.125))', 'A%B', ds=ds)
    p.close()

def test_plot_butterfly():
    "Test plot.topo.butterfly"
    ds = datasets.get_rand(utsnd=True)
    p = plot.topo.butterfly('utsnd', ds=ds)
    p.close()
    p = plot.topo.butterfly('utsnd', 'A%B', ds=ds)
    p.close()

def test_plot_array():
    "Test plot.topo.array"
    ds = datasets.get_rand(utsnd=True)
    p = plot.topo.array('utsnd', ds=ds)
    p.close()
    p = plot.topo.array('utsnd', 'A%B', ds=ds)
    p.close()
