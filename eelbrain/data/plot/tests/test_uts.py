'''
Created on Dec 2, 2012

@author: christian
'''
from eelbrain.data import factor, datasets, plot, testnd


def test_stat():
    "test plot.uts.stat plotting function"
    ds = datasets.get_rand()
    plot.uts.stat('uts', ds=ds)
    plot.uts.stat('uts', 'A%B', ds=ds)
    plot.uts.stat('uts', 'A', Xax='B', ds=ds)


def test_uts():
    "test plot.uts.uts plotting function"
    ds = datasets.get_rand()
    plot.uts.uts('uts', ds=ds)
    plot.uts.uts('uts', 'A%B', ds=ds)


def test_clusters():
    "test plot.uts cluster plotting functions"
    ds = datasets.get_rand()

    A = ds['A']
    B = ds['B']
    Y = ds['uts']

    # fixed effects model
    res = testnd.cluster_anova(Y, A * B)
    plot.uts.clusters(res, title="Fixed Effects Model")

    # random effects model:
    subject = factor(range(15), tile=4, random=True, name='subject')
    res = testnd.cluster_anova(Y, A * B * subject)
    plot.uts.clusters(res, title="Random Effects Model")

    # plot stat
    p = plot.uts.stat(Y, A % B, match=subject)
    p.plot_clusters(res.clusters[A])
    plot.uts.stat(Y, A, Xax=B, match=subject)
