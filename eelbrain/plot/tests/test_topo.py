'''
Created on Jul 22, 2013

@author: christian
'''
from eelbrain import datasets, plot


def test_plot_topomap():
    "Test plot.Topomap"
    plot.configure_backend(False, False)
    ds = datasets.get_rand(utsnd=True)
    topo = ds.eval('utsnd.summary(time=(0.075, 0.125))')

    p = plot.Topomap(topo, ds=ds)
    p.close()
    p = plot.Topomap(topo, ds=ds, vmax=0.2, w=2)
    p.close()
    p = plot.Topomap(topo, 'A%B', ds=ds, axw=2)
    p.close()

def test_plot_butterfly():
    "Test plot.TopoButterfly"
    plot.configure_backend(False, False)
    ds = datasets.get_rand(utsnd=True)
    p = plot.TopoButterfly('utsnd', ds=ds)
    p.close()
    p = plot.TopoButterfly('utsnd', ds=ds, vmax=0.2, w=2)
    p.close()
    p = plot.TopoButterfly('utsnd', 'A%B', ds=ds, axw=2)
    p.close()
    p = plot.TopoButterfly('utsnd', mark=[1, 2], ds=ds)
    p.close()
    p = plot.TopoButterfly('utsnd', mark=['1', '2'], ds=ds)
    p.close()

def test_plot_array():
    "Test plot.TopoArray"
    plot.configure_backend(False, False)
    ds = datasets.get_rand(utsnd=True)
    p = plot.TopoArray('utsnd', ds=ds)
    p.close()
    p = plot.TopoArray('utsnd', ds=ds, vmax=0.2, w=2)
    p.close()
    p = plot.TopoArray('utsnd', 'A%B', ds=ds, axw=4)
    p.close()
