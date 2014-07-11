# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import numpy as np
from matplotlib import pyplot as plt

from eelbrain import Factor, Var, datasets, plot


def test_barplot():
    "Test plot.uv.barplot"
    plot.configure_backend(False, False)
    ds = datasets.get_uv()
    plot.uv.Barplot('fltvar', 'A%B', match='rm', ds=ds)
    plot.uv.Barplot('fltvar', 'A%B', match='rm', test=0, ds=ds)
    plt.close('all')


def test_boxplot():
    "Test plot.uv.boxplot"
    plot.configure_backend(False, False)
    ds = datasets.get_uv()
    plot.uv.Boxplot('fltvar', 'A%B', match='rm', ds=ds)
    plt.close('all')

    # many pairwise significances
    ds['fltvar'][ds.eval("A%B==('a1','b1')")] += 1
    ds['fltvar'][ds.eval("A%B==('a2','b2')")] -= 1
    ds['C'] = Factor('qw', rep=10, tile=4)
    plot.uv.Boxplot('fltvar', 'A%B%C', ds=ds)


def test_histogram():
    "Test plot.uv.histogram"
    plot.configure_backend(False, False)
    ds = datasets.get_rand()
    plot.uv.Histogram('Y', 'A%B', ds=ds)
    plot.uv.Histogram('Y', 'A%B', match='rm', ds=ds)
    plt.close('all')


def test_scatterplot():
    "Test plot.uv.corrplot and lot.uv.regplot"
    plot.configure_backend(False, False)
    ds = datasets.get_rand()
    ds['cov'] = ds['Y'] + np.random.normal(0, 1, (60,))

    plot.uv.Correlation('Y', 'cov', ds=ds)
    plot.uv.Correlation('Y', 'cov', 'A%B', ds=ds)

    plot.uv.Regression('Y', 'cov', ds=ds)
    plot.uv.Regression('Y', 'cov', 'A%B', ds=ds)

    plt.close('all')


def test_timeplot():
    "Test plot.uv.timeplot"
    plot.configure_backend(False, False)
    ds = datasets.get_rand()
    ds['seq'] = Var(np.arange(2).repeat(30))
    plot.uv.Timeplot('Y', 'B', 'seq', match='rm', ds=ds)
    plt.close('all')

