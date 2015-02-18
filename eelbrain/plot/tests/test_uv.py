# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from nose.tools import eq_, ok_
import numpy as np
from matplotlib import pyplot as plt

from eelbrain import Factor, Var, datasets, plot


def test_barplot():
    "Test plot.Barplot"
    ds = datasets.get_uv()
    plot.Barplot('fltvar', 'A%B', match='rm', ds=ds, show=False)
    plot.Barplot('fltvar', 'A%B', match='rm', ds=ds, pool_error=False,
                 show=False)
    plot.Barplot('fltvar', 'A%B', match='rm', test=0, ds=ds, show=False)

    # Fixed top
    p = plot.Barplot('fltvar', 'A%B', ds=ds, top=2, test_markers=False,
                     show=False)
    ax = p._axes[0]
    eq_(ax.get_ylim()[1], 2)

    plt.close('all')


def test_boxplot():
    "Test plot.Boxplot"
    ds = datasets.get_uv()
    plot.Boxplot('fltvar', 'A%B', match='rm', ds=ds, show=False)
    plt.close('all')

    # many pairwise significances
    ds['fltvar'][ds.eval("A%B==('a1','b1')")] += 1
    ds['fltvar'][ds.eval("A%B==('a2','b2')")] -= 1
    ds['C'] = Factor('qw', repeat=10, tile=4)
    plot.Boxplot('fltvar', 'A%B%C', ds=ds, show=False)

    # long labels
    ds['A'].relabel({'a1': 'a very long label', 'a2': 'another very long label'})
    p = plot.Barplot('fltvar', 'A%B', ds=ds, show=False)
    labels = p._ax.get_xticklabels()
    bbs = [l.get_window_extent() for l in labels]
    for i in xrange(len(bbs) - 1):
        ok_(bbs[i].x1 < bbs[i + 1].x0)


def test_histogram():
    "Test plot.Histogram"
    ds = datasets.get_uts()
    plot.Histogram('Y', 'A%B', ds=ds, show=False)
    plot.Histogram('Y', 'A%B', match='rm', ds=ds, show=False)
    plt.close('all')


def test_scatterplot():
    "Test plot.Correlation and lot.uv.regplot"
    ds = datasets.get_uts()
    ds['cov'] = ds['Y'] + np.random.normal(0, 1, (60,))

    plot.Correlation('Y', 'cov', ds=ds, show=False)
    plot.Correlation('Y', 'cov', 'A%B', ds=ds, show=False)

    plot.Regression('Y', 'cov', ds=ds, show=False)
    plot.Regression('Y', 'cov', 'A%B', ds=ds, show=False)

    plt.close('all')


def test_timeplot():
    "Test plot.Timeplot"
    ds = datasets.get_uts()
    ds['seq'] = Var(np.arange(2).repeat(30))
    plot.Timeplot('Y', 'B', 'seq', match='rm', ds=ds, show=False)
    plt.close('all')

