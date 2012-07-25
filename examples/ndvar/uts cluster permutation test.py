import numpy as np
from eelbrain.eellab import *

T = var(np.arange(-.2, .8, .01), name='time')

# create simulated data:
# 4 conditions, 15 subjects
y = np.random.normal(0, .5, (60, len(T)))

# add an interaction effect
y[:15,20:60] += np.hanning(40) * 1
# add a main effect
y[:30,50:80] += np.hanning(30) * 1


Y = ndvar(y, dims=('case', T), name='Y')
A = factor(['a0', 'a1'], rep=30, name='A')
B = factor(['b0', 'b1'], rep=15, tile=2, name='B')


# fixed effects model
res = spm.cluster_anova(Y, A*B)
plot.uts.clusters(res, title="Fixed Effects Model")


# random effects model:
subject = factor(range(15), tile=4, random=True, name='subject')
res = spm.cluster_anova(Y, A*B*subject)
plot.uts.clusters(res, title="Random Effects Model")

# plot Y
plot.uts.stat(Y, A%B, match=subject)
