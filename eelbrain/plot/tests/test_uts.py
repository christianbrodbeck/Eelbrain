'''
Created on Dec 2, 2012

@author: christian
'''
from ...vessels  import datasets
from ...eellab import *



def test_clusters():
    "test plot.uts plotting functions"
    ds = datasets.get_basic()

    A = ds['A']
    B = ds['B']
    Y = ds['Ynd']

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
