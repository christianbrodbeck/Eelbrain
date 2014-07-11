'''
Created on Dec 2, 2012

@author: christian
'''
from eelbrain import Factor, datasets, plot, testnd


def test_stat():
    "test plot.UTSStat plotting function"
    plot.configure_backend(False, False)
    ds = datasets.get_rand()
    p = plot.UTSStat('uts', ds=ds)
    p.close()
    p = plot.UTSStat('uts', 'A%B', ds=ds)
    p.close()
    p = plot.UTSStat('uts', 'A', Xax='B', ds=ds)
    p.close()
    p = plot.UTSStat('uts', 'A%B', 'rm', sub="rm.isin(('R00', 'R01'))", ds=ds)
    p.close()

    # clusters
    sds = ds.sub("B == 'b0'")
    res = testnd.ttest_rel('uts', 'A', 'a1', 'a0', match='rm', ds=sds,
                           samples=0, pmin=0.05, mintime=0.02)
    p = plot.UTSStat('uts', 'A', clusters=res.clusters, ds=ds)
    p.close()
    res = testnd.ttest_rel('uts', 'A', 'a1', 'a0', match='rm', ds=sds,
                           samples=100, pmin=0.05, mintime=0.02)
    p = plot.UTSStat('uts', 'A', clusters=res.clusters, ds=ds)
    p.close()


def test_uts():
    "test plot.UTS plotting function"
    plot.configure_backend(False, False)
    ds = datasets.get_rand()
    p = plot.UTS('uts', ds=ds)
    p.close()
    p = plot.UTS('uts', 'A%B', ds=ds)
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
    p = plot.UTSClusters(res, title="Fixed Effects Model")
    p.close()

    # random effects model:
    subject = Factor(range(15), tile=4, random=True, name='subject')
    res = testnd.anova(Y, A * B * subject, samples=2)
    p = plot.UTSClusters(res, title="Random Effects Model")
    p.close()

    # plot UTSStat
    p = plot.UTSStat(Y, A % B, match=subject)
    p.set_clusters(res.clusters)
    p.close()
    p = plot.UTSStat(Y, A, Xax=B, match=subject)
    p.close()
