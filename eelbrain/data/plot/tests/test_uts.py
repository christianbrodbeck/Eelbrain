'''
Created on Dec 2, 2012

@author: christian
'''
from eelbrain.data import factor, datasets, plot, testnd


def test_stat():
    "test plot.uts.stat plotting function"
    plot.configure_backend(False, False)
    ds = datasets.get_rand()
    p = plot.uts.stat('uts', ds=ds)
    p.close()
    p = plot.uts.stat('uts', 'A%B', ds=ds)
    p.close()
    p = plot.uts.stat('uts', 'A', Xax='B', ds=ds)
    p.close()


def test_uts():
    "test plot.uts.uts plotting function"
    plot.configure_backend(False, False)
    ds = datasets.get_rand()
    p = plot.uts.uts('uts', ds=ds)
    p.close()
    p = plot.uts.uts('uts', 'A%B', ds=ds)
    p.close()


def test_clusters():
    "test plot.uts cluster plotting functions"
    plot.configure_backend(False, False)
    ds = datasets.get_rand()

    A = ds['A']
    B = ds['B']
    Y = ds['uts']

    # fixed effects model
    res = testnd.anova(Y, A * B)
    p = plot.uts.clusters(res, title="Fixed Effects Model")
    p.close()

    # random effects model:
    subject = factor(range(15), tile=4, random=True, name='subject')
    res = testnd.anova(Y, A * B * subject, samples=2)
    p = plot.uts.clusters(res, title="Random Effects Model")
    p.close()

    # plot stat
    p = plot.uts.stat(Y, A % B, match=subject)
    p.set_clusters(res.clusters)
    p.close()
    p = plot.uts.stat(Y, A, Xax=B, match=subject)
    p.close()
