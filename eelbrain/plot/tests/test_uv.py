# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import numpy as np
from matplotlib import pyplot as plt

from eelbrain import Factor, Var, datasets, plot


def test_barplot():
    "Test plot.Barplot"
    plot.configure_backend(False, False)
    ds = datasets.get_uv()
    plot.Barplot('fltvar', 'A%B', match='rm', ds=ds)
    plot.Barplot('fltvar', 'A%B', match='rm', ds=ds, pool_error=False)
    plot.Barplot('fltvar', 'A%B', match='rm', test=0, ds=ds)
    plt.close('all')


def test_boxplot():
    "Test plot.Boxplot"
    plot.configure_backend(False, False)
    ds = datasets.get_uv()
    plot.Boxplot('fltvar', 'A%B', match='rm', ds=ds)
    plt.close('all')

    # many pairwise significances
    ds['fltvar'][ds.eval("A%B==('a1','b1')")] += 1
    ds['fltvar'][ds.eval("A%B==('a2','b2')")] -= 1
    ds['C'] = Factor('qw', repeat=10, tile=4)
    plot.Boxplot('fltvar', 'A%B%C', ds=ds)


def test_histogram():
    "Test plot.Histogram"
    plot.configure_backend(False, False)
    ds = datasets.get_uts()
    plot.Histogram('Y', 'A%B', ds=ds)
    plot.Histogram('Y', 'A%B', match='rm', ds=ds)
    plt.close('all')


def test_scatterplot():
    "Test plot.Correlation and lot.uv.regplot"
    plot.configure_backend(False, False)
    ds = datasets.get_uts()
    ds['cov'] = ds['Y'] + np.random.normal(0, 1, (60,))

    plot.Correlation('Y', 'cov', ds=ds)
    plot.Correlation('Y', 'cov', 'A%B', ds=ds)

    plot.Regression('Y', 'cov', ds=ds)
    plot.Regression('Y', 'cov', 'A%B', ds=ds)

    plt.close('all')


def test_timeplot():
    "Test plot.Timeplot"
    plot.configure_backend(False, False)
    ds = datasets.get_uts()
    ds['seq'] = Var(np.arange(2).repeat(30))
    plot.Timeplot('Y', 'B', 'seq', match='rm', ds=ds)
    plt.close('all')

