import numpy as np
from eelbrain.lab import *
# the time dimension object
from eelbrain.data.data_obj import UTS

T = UTS(-.2, .01, 100)

# create simulated data:
# 4 conditions, 15 subjects
y = np.random.normal(0, .5, (60, len(T)))

# add an interaction effect
y[:15, 20:60] += np.hanning(40) * 1
# add a main effect
y[:30, 50:80] += np.hanning(30) * 1


Y = NDVar(y, dims=('case', T), name='Y')
A = Factor(['a0', 'a1'], rep=30, name='A')
B = Factor(['b0', 'b1'], rep=15, tile=2, name='B')


# fixed effects model
# (increase the number of samples for a more accurate result)
res = testnd.anova(Y, A * B, samples=100)
plot.UTSClusters(res, title="Fixed Effects Model")


# random effects model:
subject = Factor(range(15), tile=4, random=True, name='subject')
res = testnd.anova(Y, A * B * subject, samples=100, match=subject)
plot.UTSClusters(res, title="Random Effects Model")

# plot Y
p = plot.UTSStat(Y, A % B, match=subject, clusters=res.clusters)

# table with all clusters
print res.clusters
